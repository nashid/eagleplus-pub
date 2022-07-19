import importlib
import itertools
import multiprocessing as mp
import os
import pickle
import queue
import signal
import time
from contextlib import contextmanager

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering

import util
from constant import MAX_NUM_DIM, Status, MAX_PERMUTE
from errors import FuzzerError
from param import Param


class FuzzerFailureStatus:
    def __init__(self):
        self.consec_fail_count = 0  # used to determine if too many cases failed in expect_ok case
        self.is_consec_fail = False
        self.reduce_failure = False
        self.ok_seed_list = []
        self.excpt_seed_list = []


class Fuzzer:
    """
    For fuzzing by specified from one configuration
    """
    def __init__(self, config, data_construct=None):
        self.config = config
        self.data_construct = data_construct

        self.exit_flag = False
        self.fstatus = FuzzerFailureStatus()
        self.import_statements = []
        self.total_iter = 0
        self.tried_empty = False
        self.valid_permute_dtypes = []

        # after this, may want to check first if package is importable
        # NOTE: doing this means every Fuzzer object would try to do the import
        # this may not be the most efficient way to check if package is importable
        self._check_target_package()
        return

    def _check_target_package(self):
        mod = None
        root_mod = None
        try:
            print('Trying to import target package...')
            mod = importlib.import_module(self.config.package)  # only import at the init stage of a Fuzzer object
            root_mod = mod
            self.import_statements.append('import %s' % self.config.package)
        except ModuleNotFoundError:
            util.error('cannot import the specified module %s; need to correct in config file.' % self.config.package)

        assert mod is not None

        if mod.__version__ != self.config.version:
            util.error('version specified in config %s does not match with the one in environment %s'
                       % (self.config.version, mod.__version__))

        print('Package %s %s imported successfully' % (self.config.package, self.config.version))

        # parse title to get down to the function
        # e.g. `tf.math.sin` needs first get to the `math` module, and then `sin`
        title_toks = self.config.title.split('.')
        assert len(title_toks) > 1
        assert title_toks[-1] == self.config.target
        self.target_func_statement = '.'.join([self.config.package] + title_toks[1:])

        # go down the chain to get the target function
        for i, t in enumerate(title_toks):
            if i == 0:  # the root module (e.g. tf in case of tensorflow)
                continue
            if i < len(title_toks) - 1:  # until the second last
                try:
                    mod = getattr(mod, t)
                except AttributeError:
                    import_statement = '.'.join(title_toks[:i + 1])
                    self.import_statements.append(import_statement)
                    mod = importlib.import_module(import_statement)
            else:
                break

        # test the target function exists
        try:
            target_func = getattr(mod, self.config.target)
            self._target_func = target_func
        except AttributeError:
            util.error('target function %s does not exist in package %s' % (self.config.target, self.config.package))

        # test the data_construct
        if self.data_construct is None:
            return

        data_construct_toks = self.data_construct.split('.')
        mod = root_mod
        for t in data_construct_toks:
            try:
                mod = getattr(mod, t)
            except AttributeError:
                util.error('given data construnct %s is not found in package %s'
                           % (self.data_construct, self.config.package))
        assert mod is not None
        self.data_construct = mod

    @property
    def default_ndims(self):
        return list(np.arange(MAX_NUM_DIM))

    def add_ok_seed(self, seed_path):
        # only save the seed_path to save memory
        self.fstatus.ok_seed_list.append(seed_path)

    def add_excpt_seed(self, seed_path):
        # only save the seed_path to save memory
        self.fstatus.excpt_seed_list.append(seed_path)

    def _run_max_iter(self, max_iter, obey, gen_script=False):
        runforever = False
        if max_iter == -1:  # run indefinitely
            runforever = True
        seeds = self._check_exist_seed()
        count = 0
        while True:
            count += 1
            if count > max_iter and not runforever:
                break
            print('{:#^50}'.format(' %d/%d begin ' % (count, max_iter)))
            # making param to reset the shape_var
            Param.reset_shape_var()
            try:
                new_seed = self._generate(seeds, obey=obey, reduce_failure=self.fstatus.reduce_failure)
            except FuzzerError as e:
                util.warning('fuzzer failed to generate an input, will try again')
                print(e)
                continue
            if self.exit_flag:
                return
            if not new_seed:
                util.warning('empty input')
            self._test(new_seed, count, obey=obey, save=True, gen_script=gen_script)
            print('{:#^50}'.format(' %d/%d done ' % (count, max_iter)))
            print('{:-^50}'.format(''))

    def _run_max_time(self, max_time, obey, gen_script=False):
        def runforever():
            seeds = self._check_exist_seed()
            while True:
                new_seed = self._generate(seeds, obey=obey)
                if self.exit_flag:
                    return
                if not new_seed:
                    print('Error: Failed to generate input, will try again')
                    continue
                self._test(new_seed, obey=obey, save=True, gen_script=gen_script)

        if max_time == -1:  # run indefinitely
            runforever()
            # NOTE: if max_time == 0, anything after this is unreachable

        @contextmanager
        def timeout(sec):
            signal.signal(signal.SIGALRM, raise_timeout)
            # Schedule the signal to be sent after ``time``.
            signal.alarm(sec)
            try:
                yield
            except TimeoutError:
                pass
            finally:
                # Unregister the signal so it won't be triggered
                # if the timeout is not reached.
                signal.signal(signal.SIGALRM, signal.SIG_IGN)

        def raise_timeout(signum, frame):
            print("The time limit specified has been reached, exiting...")
            os._exit(0)

        with timeout(max_time):
            runforever()

    def run(self, obey=True, max_iter=0, max_time=0, gen_script=False):
        assert max_iter != 0 or max_time != 0

        start = time.time()
        if max_iter:
            self._run_max_iter(max_iter, obey, gen_script=gen_script)
        else:
            self._run_max_time(max_time, obey, gen_script=gen_script)
        end = time.time()

        def output_fuzzing_time(sec):
            timefile = self.config.workdir + '/fuzz_time'
            msg = '\n### Total time used for input generation and testing: %.2f sec\n' % sec
            util.output_time(timefile, sec, msg)

        output_fuzzing_time(end - start)

    def _check_exist_seed(self):
        """
        This function checks if seeds exist
        if exists, use one of the existing seeds to mutate
        TODO: consider if this func makes sense: we don't have coverage to guide mutation
        :return:
        """
        seeds = []
        return seeds

    def _generate(self, seeds, use_seeds=False, obey=True, reduce_failure=False):
        """
        This function generate a new seed according to the constraints
        :param seeds: existing seeds to generate based on TODO: consider if this is needed
        :return: new seed
        """
        if not reduce_failure:
            if use_seeds:
                return self._generate_from_existing(seeds, obey)
            return self._generate_new_input(obey)
        else:
            # requested by user OR
            # generated too much false alarms
            # need to try to generate input that isn't false alarm
            assert self.config.adapt_to == 'prev_ok' or self.config.adapt_to == 'permute'

            if self.config.adapt_to == 'prev_ok':
                return self._generate_by_ok_dtype(obey)
            # permutation
            self._verify_dtype_permute(obey)
            return self._generate_by_permute_dtype(obey)

    def _generate_by_permute_dtype(self, obey):
        if len(self.valid_permute_dtypes) == 0:
            util.warning(
                'False alarms may not be caused by incompatible dtypes; unable to generate valid input; exit...')
            self.exit_flag = True
            return
        dtype_comb = self.valid_permute_dtypes[np.random.randint(0, len(self.valid_permute_dtypes))]
        return self._gen_from_dtype_comb(obey, dtype_comb)

    def _verify_dtype_permute(self, obey):
        def get_all_params_dtypes():
            # for required parameters
            dtypes, python_types = get_all_params_dtypes_helper(self.config.required_param)
            if not self.config.fuzz_optional:
                return dtypes, python_types

            # for optional parameters
            opt_dtypes, opt_python_types = get_all_params_dtypes_helper(self.config.optional_param)
            return dtypes + opt_dtypes, python_types + opt_python_types

        def get_all_params_dtypes_helper(params_dict):
            dtypes = []
            python_types = []
            # default_dtypes corresponding python types (to differentiate from numpy correspondence)
            default_dtypes = {k for k, v in self.config.dtype_map.items() if v is not None}
            default_python_types = {util.convert_dtype_to_pytype(dtype) for dtype in default_dtypes}
            for p in params_dict.values():
                if p.valid_dtypes is None:
                    dtypes.append(default_dtypes)
                    python_types.append(default_python_types)
                else:
                    dtypes.append(p.valid_dtypes)
                    valid_dtypes_python_types = set()
                    for d in p.valid_dtypes:
                        py_type = util.convert_dtype_to_pytype(d)
                        valid_dtypes_python_types.add(py_type)
                    python_types.append(list(valid_dtypes_python_types))
            return dtypes, python_types

        def check_permutations(permute_types, param_names, num_permutes, output_fname):
            valid_dtypes = []
            count = 0
            permute_fname = self.config.workdir + '/' + output_fname
            with open(permute_fname, 'w+') as wf:
                for types in permute_types:
                    d = dict(zip(param_names, types))
                    count += 1
                    # need to test the input to verify if the dtype_comb is valid
                    print('[##    ', end='', flush=True)
                    testing_pair = ', '.join(['%s:%s' % kvalue for kvalue in zip(d.keys(), d.values())])
                    print('{:<50}'.format('testing %s' % testing_pair), end='', flush=True)
                    print('{:>20}'.format('%d/%d ##]' % (count, num_permutes)), end='', flush=True)
                    wf.write('[##    ')
                    wf.write('{:<50}'.format('testing %s' % testing_pair))
                    wf.write('{:>20}'.format('%d/%d ##]' % (count, num_permutes)))
                    valid = False
                    if self.config.verbose:
                        print()
                    try:
                        inputs = self._gen_from_dtype_comb(obey, d)
                    except FuzzerError:
                        util.warning('failed to generate input during permute')
                        print('=> No')
                        wf.write('=> No\n')
                        continue

                    verbose_ctx = util.verbose_mode(self.config.verbose)
                    save_flag = self.config.verbose
                    with verbose_ctx:
                        # if --verbose, save input and gen_script, otherwise, only save whenever Timeout/Signal occurs
                        status = self._test(inputs, obey=obey, save=save_flag, gen_script=save_flag)
                    if status == Status.PASS:  # the dtype combination was fine
                        valid_dtypes.append(d)
                        valid = True
                    elif status == Status.TIMEOUT:  # timeout error
                        print('Error: Fuzzer Timedout!! Need Investigation!!')
                        print('=> Timeout')
                        wf.write('=> Timeout\n')
                        valid_dtypes.append(d)
                        continue
                    elif status == Status.SIGNAL:  # signal
                        print('Error: Fuzzer got a Signal!! Need Investigation!!')
                        print('=> Signal')
                        wf.write('=> Signal\n')
                        valid_dtypes.append(d)
                        continue
                    # else the dtype combination made the target function throw exception
                    if valid:
                        print('=> Yes')
                        wf.write('=> Yes\n')
                    else:
                        print('=> No')
                        wf.write('=> No\n')
                assert count == num_permutes
                return valid_dtypes

        if self.valid_permute_dtypes:  # no need to redo
            return

        # for each parameter, get the possible valid_dtypes and python types
        all_param_dtypes, all_param_python_types = get_all_params_dtypes()
        # calculate the number of permutations
        num_permute_pytypes = util.calc_num_permute(all_param_python_types)
        num_permute_dtypes = util.calc_num_permute(all_param_dtypes)
        if num_permute_dtypes > MAX_PERMUTE:
            util.warning('too many permutations need to be done, will try to reduce')
            all_param_dtypes = util.reduce_dtypes(all_param_dtypes)
            num_permute_dtypes = util.calc_num_permute(all_param_dtypes)

        # get the permutations
        permute_dtypes = itertools.product(*all_param_dtypes)
        permute_python_types = itertools.product(*all_param_python_types)

        # reconstruct key-value dict for each: {param1: dtype1, param2: dtype2, ...}
        param_names = list(self.config.required_param.keys())
        if self.config.fuzz_optional:
            param_names += list(self.config.optional_param.keys())

        print('>>>>>> checking python default types...')
        valid_python_types = check_permutations(permute_python_types, param_names,
                                                num_permute_pytypes, 'py_types_permute')
        print('>>>>>> checking valid_dtypes...')
        valid_dtypes = check_permutations(permute_dtypes, param_names,
                                          num_permute_dtypes, 'dtype_permute')

        # get the set of dtypes that may behave inconsistently from default python types
        incon_dtypes = self._crosscheck_types(valid_python_types, valid_dtypes)

        # separate the truly valid dtypes and the questionable ones
        true_valid_dtypes = []
        question_valid_dtypes = []
        for dt in valid_dtypes:
            tup = tuple(dt.values())
            if tup in incon_dtypes:
                question_valid_dtypes.append(dt)
            else:
                true_valid_dtypes.append(dt)

        self.valid_permute_dtypes = true_valid_dtypes + valid_python_types
        self.question_valid_dtypes = question_valid_dtypes
        return

    def _crosscheck_types(self, valid_python_types, valid_dtypes):
        """
        check if valid_dtypes is also valid in python types
        e.g. if dtype combination of (float32, float32, int32) is valid
        then, (float, float, int) should also be valid
        :param valid_python_types: list of dict in form {param1: py_type1, param2: py_type2, ...}
        :param valid_dtypes: list of dict in form {param1: dtype1, param2: dtype2, ...}
        :return:
        """
        # each element of the set is a tuple of python types
        python_type_set = set([tuple(t.values()) for t in valid_python_types])

        incon_dtypes = []
        for dt in valid_dtypes:
            tup = tuple(map(util.convert_dtype_to_pytype, dt.values()))
            if tup not in python_type_set:
                incon_dtypes.append(tuple(dt.values()))
                print(incon_dtypes[-1], end='')
                print(' is valid <--> ', end='')
                print(tup, end='')
                print(' is invalid')
        return set(incon_dtypes)

    def _pick_disobey_param(self, obey):
        if obey:
            return []

        # violating constraint
        # Note: it's not interesting to make all parameters to violate constraints
        # because one parameter violating constraints will lead to the entire function call fail
        # so, it's more interesting to make one parameter violate at a time
        disobey_candidates_required_param = [p.name for p in self.config.required_param.values() if p.can_disobey]
        disobey_candidates_optional_param = [p.name for p in self.config.optional_param.values() if p.can_disobey]
        disobey_all = disobey_candidates_required_param + disobey_candidates_optional_param
        if self.config.fuzz_optional:
            pnames = [pname for pname in disobey_all]
        else:  # only fuzz required params
            pnames = [pname for pname in disobey_candidates_required_param]

        if pnames:  # possible to disobey constraints
            disobey_param = list(np.random.choice(pnames, size=1))  # pick 1 random param to disobey
            util.DEBUG('selected %s to violate constraints' % disobey_param)
            return disobey_param

        util.warning('no parameter can disobey constraints, falling back to random generation')
        return []

    def _gen_from_dtype_comb(self, obey, dtype_comb={}):
        disobey_param = self._pick_disobey_param(obey)

        def gen_from_dtype_helper(name, param_obey):
            chosen_dtype = dtype_comb.get(name)
            if name in self.config.required_param:
                p_obj = self.config.required_param[name]
            else:
                p_obj = self.config.optional_param[name]
            return p_obj.gen(param_obey, self.default_ndims, self.config.dtype_map,
                             self.data_construct, chosen_np_dtype=chosen_dtype)

        gen_inputs = {}
        for pname in self.config.gen_order:
            if not self.config.fuzz_optional and pname in self.config.optional_param:
                continue
            if pname in disobey_param:
                p_obey = False
            else:
                p_obey = True
            gen_inputs[pname] = gen_from_dtype_helper(pname, p_obey)
        return gen_inputs

    def _generate_by_ok_dtype(self, obey):
        # NOTE: the dtype combination here are in numpy dtypes
        ok_dtypes = self._get_ok_seed_dtype_pattern()  # in form [{var1: dtype1, var2: dtype2}, {...}, ]
        # pick a dtype comb
        dtype_comb = ok_dtypes[np.random.randint(0, len(ok_dtypes))]
        print('*** Trying to generate from previously OK dtypes: ', dtype_comb)
        return self._gen_from_dtype_comb(obey, dtype_comb)

    def _get_ok_seed_dtype_pattern(self):
        if len(self.fstatus.ok_seed_list) == 0:  # empty
            return [{}]

        ok_dtype = []
        dtype_record = set()  # used to keep track if dtype combination is already added
        for each_path in self.fstatus.ok_seed_list:  # each file is a pickle file
            assert each_path.endswith('.p')
            with open(each_path, 'rb') as f:  # TODO: this may create too much diskIO, consider to refactor
                data = pickle.load(f)
                dtypes = {}
                dtype_tup = []
                for name, value in data.items():
                    if hasattr(value, 'dtype'):
                        dtypes[name] = str(value.dtype)
                    else:
                        # for built-in data type: bool, str
                        # TODO: better way to get the type string?
                        dtype_keyword = str(type(value)).split("'")[1]
                        if '.' in dtype_keyword:
                            dtype_keyword = dtype_keyword.split('.')[1]
                        dtypes[name] = dtype_keyword
                    dtype_tup.append(dtypes[name])
                dtype_tup = tuple(dtype_tup)
                # check if this dtypes combination already exists
                if dtype_tup in dtype_record:
                    continue
                dtype_record.add(dtype_tup)
                ok_dtype.append(dtypes)
        return ok_dtype

    def _generate_from_existing(self, seeds, obey):
        """
        This function generates new input by mutating existing seed
        The logic could involve a look up table to check which seed has already
        been mutated and so choose another to mutate
        :param seeds: existing seeds
        :return: new seed
        """
        # TODO
        raise NotImplementedError

    def _generate_new_input(self, obey):
        gen_inputs = {}

        def need_to_try_empty_input():
            # if all inputs are optional & expect_ok => try empty input first
            if obey and not self.config.required_param and not self.tried_empty:
                return True
            # if at least one required input & expect exception => try empty input first
            if not obey and self.config.required_param and not self.tried_empty:
                return True
            return False

        if need_to_try_empty_input():
            self.tried_empty = True
            print('-- Try Empty Input --')
            return gen_inputs

        return self._gen_from_dtype_comb(obey)

    # def _test(self, seed, obey=True, save=True, gen_script=False):
    #     """
    #     This function tests the target function with seed
    #     :return: 0 if the test was fine; 1 if the test had issue; 2 if the test timeout
    #     """
    #     # we first save the input
    #     # so even when target program crashes, we still know which input caused it
    #     if save:
    #         seed_fpath, is_duplicate = util.save_seed_to_file(seed, self.config.workdir)
    #     else:
    #         seed_fpath, is_duplicate = '', False
    #
    #     self.total_iter += 1
    #
    #     if is_duplicate:
    #         util.warning('skipping a duplicate')
    #         return Status.PASS
    #
    #     # if we want to generate a python script to reproduce the generated input
    #     test_script_path = util.gen_input_test_script(seed_fpath,
    #                                                   self.import_statements,
    #                                                   self.target_func_statement) if gen_script else ''
    #
    #     exception_list = self.config.exceptions
    #
    #     worker_func = self._expect_ok if obey else self._expect_exception
    #
    #     res_queue = mp.Queue(1)
    #     p = mp.Process(target=worker_func, args=(seed, res_queue, seed_fpath, exception_list))

    #     p.start()
    #     try:
    #         self.fstatus, status = res_queue.get(timeout=self.config.timeout)
    #         assert status == Status.PASS or status == Status.FAIL
    #         if save and status == Status.FAIL:
    #             # failure: either (expect_ok & got exception) or (expect_exception & no exception)
    #             util.report_failure(seed_fpath, test_script_path, self.config.workdir)
    #         return status
    #     except queue.Empty:  # exception caused by timeout
    #         util.warning('a Timeout occurred')
    #     finally:
    #         p.join(timeout=.01)
    #         if p.is_alive() or p.exitcode is None:
    #             p.terminate()
    #             p.join()
    #
    #     # either Timeout or Signal error
    #     # in case `save` == False, save the input and the script
    #     seed_fpath, is_duplicate = util.save_seed_to_file(seed, self.config.workdir)
    #     test_script_path = util.gen_input_test_script(seed_fpath,
    #                                                   self.import_statements,
    #                                                   self.target_func_statement)
    #     if p.exitcode < 0 and -p.exitcode != signal.SIGTERM:
    #         util.report_signal_error_input(p.exitcode, seed_fpath, test_script_path, self.config.workdir)
    #         return Status.SIGNAL
    #
    #     # timeout_error
    #     if -p.exitcode != signal.SIGTERM:  # TODO: remove this debug block later when problem is clear
    #         debug_log_fname = self.config.workdir + '/debug.log'
    #         with open(debug_log_fname, 'w+') as f:
    #             f.write('# this file is to record a timeout that had exitcode other than SIGTERM\n')
    #             f.write('-------------------------\n')
    #             f.write('input = %s\n' % seed_fpath)
    #             f.write('exit code = %d\n' % -p.exitcode)
    #             f.write('-------------------------\n')
    #         exit(1)
    #
    #     util.report_timeout_error_input(seed_fpath, test_script_path, self.config.workdir)
    #     return Status.TIMEOUT

    def _test(self, seed, count=None, obey=True, save=True, gen_script=False):
        """
        This function tests the target function with seed
        :return: 0 if the test was fine; 1 if the test had issue; 2 if the test timeout
        """
        if save:
            seed_fpath, is_duplicate = util.save_seed_to_file(seed, self.config.workdir, count)
        else:
            seed_fpath, is_duplicate = '', False

        self.total_iter += 1

        if is_duplicate:
            util.warning('skipping a duplicate')
            return Status.PASS

        # if we want to generate a python script to reproduce the generated input
        test_script_path = util.gen_input_test_script(seed_fpath,
                                                      self.import_statements,
                                                      self.target_func_statement) if gen_script else ''

        exception_list = self.config.exceptions

        worker_func = self._expect_ok if obey else self._expect_exception

        res_queue = mp.Queue(1)
        p = mp.Process(target=worker_func, args=(seed, res_queue, seed_fpath, exception_list))

        p.start()

        def check_process(process, tlimit):
            start_time = time.time()
            while time.time() - start_time < tlimit:
                try:
                    self.fstatus, res_status = res_queue.get(timeout=0.1)
                    assert res_status == Status.PASS or res_status == Status.FAIL
                    if save and res_status == Status.FAIL:
                        # failure: either (expect_ok & got exception) or (expect_exception & no exception)
                        util.report_failure(seed_fpath, test_script_path, self.config.workdir)
                    break
                except queue.Empty:
                    if not process.is_alive():
                        res_status = Status.SIGNAL
                        break
            else:  # time limit reached
                res_status = Status.TIMEOUT
                process.terminate()

            process.join()
            return res_status

        status = check_process(p, self.config.timeout)

        if status == Status.PASS or status == Status.FAIL:
            return status

        # either Timeout or Signal error
        # in case `save` == False, save the input and the script
        seed_fpath, is_duplicate = util.save_seed_to_file(seed, self.config.workdir, count)
        test_script_path = util.gen_input_test_script(seed_fpath,
                                                      self.import_statements,
                                                      self.target_func_statement)
        if p.exitcode < 0 and -p.exitcode != signal.SIGTERM:
            util.report_signal_error_input(p.exitcode, seed_fpath, test_script_path, self.config.workdir)
            return Status.SIGNAL

        # timeout_error
        if -p.exitcode != signal.SIGTERM:  # TODO: remove this debug block later when problem is clear
            debug_log_fname = self.config.workdir + '/debug.log'
            with open(debug_log_fname, 'w+') as f:
                f.write('# this file is to record a timeout that had exitcode other than SIGTERM\n')
                f.write('-------------------------\n')
                f.write('input = %s\n' % seed_fpath)
                f.write('exit code = %d\n' % -p.exitcode)
                f.write('-------------------------\n')
            exit(1)

        util.report_timeout_error_input(seed_fpath, test_script_path, self.config.workdir)
        return Status.TIMEOUT

    def confirm_timeout(self, seed, obey, seed_fpath):
        """
        basically do the testing again with a bigger timeout limit
        :param seed: the input to be tested
        :param obey: flag to indicate whether to follow constraints
        :param seed_fpath: the input seed path
        :return: status of the testing
        """
        new_timeout = min(self.config.timeout * 10, 1800)  # 10x the specified timeout or .5h whichever the less
        util.warning("#### Going to retry the timeout with %d sec" % new_timeout)

        exception_list = self.config.exceptions

        worker_func = self._expect_ok if obey else self._expect_exception

        test_script_path = util.gen_input_test_script(seed_fpath,
                                                      self.import_statements,
                                                      self.target_func_statement)
        res_queue = mp.Queue(1)
        p = mp.Process(target=worker_func, args=(seed, res_queue, seed_fpath, exception_list))

        p.start()

        def check_process(process, tlimit):
            start_time = time.time()
            while time.time() - start_time < tlimit:
                try:
                    self.fstatus, res_status = res_queue.get(timeout=0.1)
                    assert res_status == Status.PASS or res_status == Status.FAIL
                    if res_status == Status.FAIL:
                        # failure: either (expect_ok & got exception) or (expect_exception & no exception)
                        util.report_failure(seed_fpath, test_script_path, self.config.workdir)
                    util.warning('a Timeout disappeared')
                    break
                except queue.Empty:
                    if not process.is_alive():
                        res_status = Status.SIGNAL
                        break
            else:  # time limit reached
                res_status = Status.TIMEOUT
                process.terminate()
                util.warning('a Timeout occurred')

            process.join()
            return res_status

        status = check_process(p, new_timeout)
        if status == Status.PASS or status == Status.FAIL:
            return status

        # signal error
        if p.exitcode < 0 and -p.exitcode != signal.SIGTERM:
            test_script_path = util.gen_input_test_script(seed_fpath,
                                                          self.import_statements,
                                                          self.target_func_statement)
            util.report_signal_error_input(p.exitcode, seed_fpath, test_script_path, self.config.workdir)
            return Status.SIGNAL

        # Timeout error
        return Status.TIMEOUT

    def rerun_timeout_input(self, obey):
        # read the timeout_record
        timeout_record = '/'.join([self.config.workdir, 'timeout_record'])
        confirmed_timeout_record = timeout_record + '_confirmed'

        if not os.path.exists(timeout_record):  # no timeout to re-run
            return

        with open(timeout_record, 'r') as f:
            timeout_inputs = f.read().splitlines()

        print("##### re-run the the timeout input ####")
        print("--- %d timeout input need to re-run ---" % len(timeout_inputs))
        for idx, input_i in enumerate(timeout_inputs):
            print("[%d/%d] testing %s --" % (idx+1, len(timeout_inputs), input_i))
            data = pickle.load(open(input_i, 'rb'))
            confirm_status = self.confirm_timeout(data, obey, input_i)
            if confirm_status == Status.TIMEOUT:
                with open(confirmed_timeout_record, 'a') as outf:
                    outf.write(input_i + '\n')
        if os.path.exists(confirmed_timeout_record):  # timeout_record_confirmed is only created if input timeout again
            os.rename(confirmed_timeout_record, timeout_record)
            util.sync_input_script(timeout_record)
        else:
            os.remove(timeout_record)  # all timeout input won't timeout if given more time, so delete
            os.remove(timeout_record.replace('record', 'script_record'))  # also delete the timeout_script_record

    def _expect_ok(self, seed, res_queue, seed_fpath='', exception_list=[]):
        # NOTE: although we expect status to be ok, the constraints could be very loose
        # so even though we think we might have generated valid input according to constraints
        # might still cause exceptions
        verbose_ctx = util.verbose_mode(self.config.verbose)
        try:
            with verbose_ctx:
                self._target_func(**seed)
            if seed_fpath:  # seed_fpath == '' when we don't want to save the input
                self.add_ok_seed(seed_fpath)
            else:
                res_queue.put((self.fstatus, Status.Pass))
                return
            self.fstatus.is_consec_fail = False
            self.fstatus.consec_fail_count = 0
            res_queue.put((self.fstatus, Status.PASS))
            return
        except Exception as e:
            # NOTE: with current setting, due to lack of detailed documentation
            # exceptions frequently occur in this case (especially on what valid data type should be)
            # MAY WANT TO consider to ADD another mode of execution to only care about sys signals
            print('FuzzerMessage: expects function execution to be ok, but exception occurred.')
            print('------------ Exception Message ------------')
            print(e)
            print('-------------------------------------------')

            if not seed_fpath:  # when we don't want to save the seed
                res_queue.put((self.fstatus, Status.FAIL))
                return
            # save exception message to file
            assert seed_fpath.endswith('.p')
            e_fpath = seed_fpath[:-2] + '.e'
            # only writes if not exist
            if not os.path.exists(e_fpath):
                with open(e_fpath, 'w+') as wf:
                    wf.write(str(e))

            self.add_excpt_seed(seed_fpath)
            self.fstatus.consec_fail_count += 1
            if self.fstatus.is_consec_fail:
                if self.fstatus.consec_fail_count > self.config.consec_fail:
                    util.warning('max consecutive failure reached, will try to reduce failure')
                    self.fstatus.reduce_failure = True
            else:
                self.fstatus.is_consec_fail = True
            res_queue.put((self.fstatus, Status.FAIL))
            return

    def _expect_exception(self, seed, res_queue, seed_fpath='', exception_list=[]):
        had_exception = False
        verbose_ctx = util.verbose_mode(self.config.verbose)
        try:
            with verbose_ctx:
                self._target_func(**seed)
        except Exception as e:
            # had_exception = True
            if len(exception_list) == 0:  # we don't know the valid exceptions, so we treat any exception as valid
                res_queue.put((self.fstatus, Status.PASS))
                return
            if type(e).__name__ in exception_list:  # in the valid list, also ok
                res_queue.put((self.fstatus, Status.PASS))
                return
            # not in the valid list
            # might be an error but it could be caused by insufficient knowledge of valid exceptions
            util.warning('Caught exception %s is not in the valid exception list, but may not be an error.'
                         % type(e).__name__)
            res_queue.put((self.fstatus, Status.PASS))
            return

        if not had_exception:
            print('Error: expected to have exception but got no exception')
            res_queue.put((self.fstatus, Status.FAIL))
            return

    def cluster_exceptions(self, metric, threshold):
        exception_msgs, exception_fnames = util.get_exception_messages(self.config.workdir)

        if len(exception_msgs) < 2:
            util.warning("Not enough exception messages to cluster. Clustering will not run.")
            return

        print('>>>> Trying to cluster exception messages <<<<')

        start = time.time()
        exception_msgs_X = np.array(exception_msgs).reshape(-1, 1)
        # NOTE: the metric choice has been validated at the beginning
        # so invalid metric choice shouldn't exist
        if metric == 'levenshtein':
            distance = util.levenshtein
        elif metric == 'jaccard':
            distance = util.jaccard_dist
        else:  # this shouldn't happen
            print("Fuzzer Impl Error: inconsistent metric choice implementation")
            exit(1)

        # compute the distance between the exception messages
        dist_matrix = squareform(pdist(exception_msgs_X, lambda x, y: distance(x[0], y[0])))

        # determining the number of clusters (K) is a whole different problem
        # a method to avoid defining K is by hierarchical clustering
        clusters = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                           linkage='single', distance_threshold=threshold).fit(dist_matrix)

        cluster_dict = {}
        for c_idx in range(clusters.n_clusters_):
            cluster_dict[c_idx] = []

        for i, label in enumerate(clusters.labels_):
            cluster_dict[label].append(exception_fnames[i])

        end = time.time()

        def remove_cluster_output_files():
            util.warning('removing existing cluster output files...')
            for file in [f for f in os.listdir(self.config.workdir) if f.startswith('cluster_')]:
                os.remove(self.config.workdir + '/' + file)

        remove_cluster_output_files()

        print("==> Exceptions Message Clusters <==")
        for c_idx in sorted(cluster_dict.keys()):
            # print && save cluster info to file
            cluster_fname = self.config.workdir + '/cluster_' + str(c_idx)
            with open(cluster_fname, 'w+') as wf:
                print('====== Cluster %d ======' % c_idx)
                for name in cluster_dict[c_idx]:
                    print('     %s' % name)
                    wf.write(name + '\n')

        def output_cluster_time(sec):
            timefile = self.config.workdir + '/cluster_time'
            msg = '\n### Total time used for exception clustering: %.2f sec\n' % sec
            util.output_time(timefile, sec, msg)

        output_cluster_time(end - start)
