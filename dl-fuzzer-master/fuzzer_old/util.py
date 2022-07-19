"""
utility functions to aid the fuzzer
"""
import hashlib
import os
import pickle
import re
import signal
import sys
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext

import numpy as np
from ruamel.yaml import YAML

from errors import FuzzerError


def output_time(fpath, sec, msg):
    print(msg)
    with open(fpath, 'w+') as f:
        f.write('%.2f sec\n' % sec)


def reduce_dtypes(dtypes_list):
    """
    reduce the dtypes to avoid too many permutations
    the idea is to only preserve one dtype for each python type
    :param dtypes_list: each element of the list is another list containing dtype choices for a param
    :return: the reduced list of dtypes
    """
    # TODO: need better default for each, as some default may not be common
    reduced_dtypes_list = []
    for dtype_choices in dtypes_list:
        pytypes = set()
        less_choices = []
        for d in dtype_choices:
            py_t = convert_dtype_to_pytype(d)
            if py_t in pytypes:
                continue
            pytypes.add(py_t)
            less_choices.append(d)
        reduced_dtypes_list.append(less_choices)
    return reduced_dtypes_list


def calc_num_permute(lst):
    # each element of the given list is another list contains possible choices
    product = np.prod([len(choices) for choices in lst])
    return product


def convert_dtype_to_pytype(dtype):
    if 'int' in dtype or 'short' in dtype or 'long' in dtype:
        return 'int'
    if 'float' in dtype or 'double' in dtype or 'half' in dtype:
        return 'float'
    if 'complex' in dtype:
        return 'complex'
    if 'str' in dtype:
        return 'str'
    if 'bool' in dtype:
        return 'bool'
    if 'dtype:' in dtype:  # dtype dependency
        return dtype
    if 'dtype' in dtype:
        return 'dtype'
    error('unknown dtype: %s' % dtype)


def read_yaml(fpath):
    yaml = YAML()
    try:
        f = open(fpath)
        y = yaml.load(f)
    except FileNotFoundError:
        print('FuzzerError: given input config %s not found, exiting...' % fpath)
        exit(1)
    return y


def assert_in(x, collection, x_name=None, collection_name=None):
    """
    A until function for assertion
    :param x: target to be checked in collection
    :param collection: this collection should contain x
    :param x_name: used for error message to refer to x
    :param collection_name: used for error message to refer to collection
    :return: None
    """
    err_msg = ''
    if x_name:
        err_msg += x_name
    else:
        err_msg += x
    err_msg += " doesn't exist in "
    if collection_name:
        err_msg += collection_name
    else:
        err_msg += collection
    assert x in collection, err_msg


def warning(msg):
    """
    A until function to print warning message
    :param msg: the warning message to be displayed
    :return: None
    """
    print('FuzzerWarning: %s' % msg)
    return


def add_to_file(fpath, content, mode):
    with open(fpath, mode) as f:
        f.write(content)


def save_seed_to_file(seed, workdir, gen_id):
    is_dup = False
    fname = hashlib.sha1(str(seed).encode('utf-8')).hexdigest() + '.p'
    fpath = workdir + '/' + fname
    if os.path.exists(fpath):
        is_dup = True
        return fpath, is_dup
    print('============================================')
    print('Saving seed to file %s ' % fpath)
    print('============================================')
    pickle.dump(seed, open(fpath, 'wb'))
    # record the generation order
    gen_order_file = os.path.join(workdir, 'gen_order')
    if os.path.exists(gen_order_file):
        add_to_file(gen_order_file, '%s, %s\n' % (fpath, gen_id), 'a')
    else:
        add_to_file(gen_order_file, '%s, %s\n' % (fpath, gen_id), 'w')
    return fpath, is_dup


@contextmanager
def block_out():
    devnull = open(os.devnull, 'w')
    orig_stdout_fno = os.dup(sys.stdout.fileno())
    orig_stderr_fno = os.dup(sys.stderr.fileno())
    os.dup2(devnull.fileno(), 1)  # stdout
    os.dup2(devnull.fileno(), 2)  # stderr
    try:
        yield
    finally:
        os.dup2(orig_stdout_fno, 1)
        os.close(orig_stdout_fno)
        os.dup2(orig_stderr_fno, 2)
        os.close(orig_stderr_fno)
        devnull.close()


def verbose_mode(switch):
    if switch:
        return nullcontext()
    return block_out()


def gobble_until(start_idx, text, chars=''):
    if len(text) <= start_idx:
        error('unable to process "%s"; likely an incorrect input format.' % text)

    s = ''
    for i in range(start_idx, len(text)):
        if text[i] == ' ':
            continue
        if text[i] in chars:
            break
        s += text[i]
    return s, i + 1


def check_shape_validity(shape):
    for value in shape:
        if not isinstance(value, (int, np.integer)) or value < 0:
            error('value %s in shape %s is invalid' % (str(value), str(shape)))
    return len(shape)


def gobble_var_dep(tok, sp_str=''):
    """
    the shape of a variable is dependent on the length of another variable
    e.g. [len:&dim] means the shape should be the length of the var 'dim'
    the preceeding '&' before a name means that name is a var
    Also could be [len:dim] means 'dim' is not an input variable
    :param tok: the token to be parsed
    :param sp_str: special string for delimiter
    :return:
    """
    if sp_str != '':
        tok = tok.split(sp_str)[1]

    is_var = False
    ret_ref = ''
    for i, c in enumerate(tok):
        if c == '&':
            is_var = True
            continue
        elif c in '+-*/':
            i -= 1
            break
        elif c == '_':
            pass
        elif not c.isalnum():
            error('invalid spec %s' %tok)
        ret_ref += c
    tok = tok[i+1:]
    return tok, ret_ref, is_var


def parse_shape_bound(tok):
    """
    parse the >, >=, <, <= signs
    e.g. [>=5] means a list with min length 5
    :param tok: the token to be parsed, e.g. '>=5'
    :return: sign, bound (non-inclusive)
    """
    # will force to confirm to one standard
    assert len(tok) > 1
    start = 0
    if tok[start] == '>':
        sign = '>'
        start += 1
        if tok[start] == '=':
            num, _ = gobble_until(start + 1, tok)
            bound = int(num) - 1
        else:
            num, _ = gobble_until(start, tok)
            bound = int(num)
    elif tok[start] == '<':
        sign = '<'
        start += 1
        if tok[start] == '=':
            num, _ = gobble_until(start + 1, tok)
            bound = int(num) + 1
        else:
            num, _ = gobble_until(start, tok)
            bound = int(num)
    return sign, bound


def gobble_unequal_sign(tok):
    """
    parse the >, >=, <, <= signs
    e.g. [>=5] means a list with min length 5
    NOTE: does not support composition of other operators
    such as [>=len:&a]
    :param tok: the token to be parsed, e.g. '>=5'
    :return:
    """
    if not tok or len(tok) < 2:
        error('invalid spec %s' % tok)
    assert tok[0] == '>' or tok[0] == '<'
    if tok[1] == '=':
        if len(tok) <= 2:
            error('expect a number after %s but none was provided' % tok)
        num, _ = gobble_until(2, tok)
    else:
        num, _ = gobble_until(1, tok)
    if not num.isnumeric():
        _, ref, is_var = gobble_var_dep(num)
    else:
        ref, is_var = None, False
    return ref, is_var


def pick_scalar(dtype_list):
    # only from real numbers
    signed_ints = pick_integer(dtype_list)
    unsigned_ints = pick_integer(dtype_list, unsigned=True)
    floats = pick_float(dtype_list)
    return signed_ints + unsigned_ints + floats


def pick_integer(dtype_list, unsigned=False):
    signed_list = []
    unsigned_list = []
    for dtype in dtype_list:
        if re.search('.*int[0-9]*', dtype):
            # we follow the definition of integer to be signed int, so further filtering
            if 'u' in dtype and unsigned:
                unsigned_list.append(dtype)
            else:
                signed_list.append(dtype)
    return unsigned_list if unsigned else signed_list


def pick_float(dtype_list):
    res_list = []
    for dtype in dtype_list:
        if re.search('.*float[0-9]*', dtype):
            res_list.append(dtype)
    return res_list


def is_power_two(n):
    return (n & (n - 1) == 0) and n != 0


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_dtype_compatible(dtype1, dtype2):
    if (dtype1 and dtype2) is None:  # Note: cannot be both None, so one of them is None
        return False

    dtypes_list = ['int', 'float', 'str', 'bool', 'complex', 'dtype', 'shape']
    if any(d in dtype1 and d in dtype2 for d in dtypes_list):
        return True
    return False


def is_ndim_compatible(ndim1, ndim2):
    return ndim1 == ndim2


def validate_kwargs(kwargs, allowed_kwargs, error_message='Keyword argument not understood:'):
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError(error_message, kwarg)


def error(msg):
    raise FuzzerError(msg)


def record_special_input(workdir, filename, content):
    record_path = '/'.join([workdir, filename])
    if not os.path.exists(record_path):
        with open(record_path, 'w+') as f:
            f.write(content + '\n')
        return

    with open(record_path, 'r+') as f:
        for l in f:
            l.replace('\n', '')  # remove the newline
            if content in l:
                break
        else:
            f.write(content + '\n')


def report_signal_error_input(exitcode, seed_fpath, test_script_path, workdir):
    assert exitcode < 0
    code = -exitcode

    def report_signal(signal_name):
        print('\n#### Received %s signal\n' % signal_name)
        signal_name = signal_name.replace(' ', '_')
        signal_input_record = signal_name + '_record'
        record_special_input(workdir, signal_input_record, seed_fpath)
        signal_script_record = signal_name + '_script_record'
        record_special_input(workdir, signal_script_record, 'python ' + test_script_path)

    if code == signal.SIGABRT:
        report_signal('Abort')
    elif code == signal.SIGBUS:
        report_signal('Bus Error')
    elif code == signal.SIGFPE:
        report_signal('Floating Point Exception')
    elif code == signal.SIGILL:
        report_signal('Illegal instruction')
    elif code == signal.SIGKILL:
        report_signal('Kill')
    elif code == signal.SIGPIPE:
        report_signal('Broken Pipe')
    elif code == signal.SIGSEGV:
        report_signal('Segmentation Fault')
    else:
        warning('Un-handled signal %d' % code)


def report_timeout_error_input(seed_fpath, test_script_path, workdir):
    warning('Process Timed Out, continue, but there might be a serious problem with the input')
    record_special_input(workdir, 'timeout_record', seed_fpath)
    record_special_input(workdir, 'timeout_script_record', 'python ' + test_script_path)


def report_failure(seed_fpath, test_script_path, workdir):
    if seed_fpath == '':
        warning('received empty path for the generated input, will not record')
    else:
        record_special_input(workdir, 'failure_record', seed_fpath)
    if test_script_path == '':
        warning('received empty path for the generated test script, will not record')
    else:
        record_special_input(workdir, 'failure_script_record', 'python ' + test_script_path)


def gen_input_test_script(seed_fpath, import_statements, target_func_statement):
    """
    generate a python script to load the generated input and do the function call
    :param seed_fpath: the seed file path
    :param import_statements: the list of import statements needed to run target_func_statement
    :param target_func_statement: the statement to execute the target function
    :return: the path to the generated script
    """
    if not seed_fpath:
        return ''
    assert seed_fpath.endswith('.p')
    assert isinstance(import_statements, list)
    assert isinstance(target_func_statement, str)
    script_path = seed_fpath[:-2] + '.py'
    if os.path.exists(script_path):  # won't generate again
        return script_path
    with open(script_path, 'w+') as wf:
        wf.write('import pickle\n')  # used for loading the generated input
        for s in import_statements:
            wf.write(s + '\n')
        wf.write('data = pickle.load(open(\'%s\', \'rb\'))\n' % seed_fpath)
        wf.write(target_func_statement + '(**data)\n')
    return script_path


def sync_input_script(fpath):
    """
    synchronize content in timeout_script_record and timeout_record
    :param fpath: path to timeout_record
    :return:
    """
    with open(fpath, 'r') as f:
        input_paths = f.read().splitlines()
    script_record = fpath.replace('record', 'script_record')

    with open(script_record, 'r') as f:
        script_record_lines = set(f.read().splitlines())
    final_script_record = []
    for path in input_paths:
        assert path.endswith('.p')
        target_line = 'python ' + path + 'y'
        if target_line in script_record_lines:  # note: each line in script_record is in form 'python xxx.py'
            final_script_record.append(target_line)

    with open(script_record, 'w+') as f:
        f.write('\n'.join(final_script_record))
        f.write('\n')


def str_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def str_is_float(s):
    if str_is_int(s):
        return False
    return s.replace('-', '').replace('.', '', 1).isdigit()


def str_is_inf(s):
    return s in ['inf', '-inf']


def str_is_number(s):
    return str_is_int(s) or str_is_float(s) or str_is_inf(s)


def str_is_bool(s):
    if s.lower() in ['true', 'false']:
        return True
    return False


def convert_to_num(s):
    if str_is_int(s):
        return int(s)
    elif str_is_float(s):
        return float(s)
    elif s == 'inf':
        return np.inf
    elif s == '-inf':
        return np.NINF
    else:
        error('failed to convert "%s" to a number' % s)


def infer_type_from_str(s):
    if str_is_int(s):
        return 'int'
    elif str_is_float(s):
        return 'float'
    elif str_is_bool(s):
        return 'bool'
    return 'str'


def try_cast(val, cast_type):
    if cast_type == 'int':
        cast_func = int
    elif cast_type == 'float':
        cast_func = float
    elif cast_type == 'complex':
        cast_func = complex
    elif cast_type == 'str':
        cast_func = str
    else:
        error('unknown cast type %s' % cast_type)

    try:
        return cast_func(val)
    except ValueError:
        error('unable to cast value %s to type %s' % (str(val), cast_type))


def levenshtein(s1, s2):
    # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def jaccard_dist(s1, s2):
    # s1, s2 are 2 strings
    s1_set = set(s1.split())
    s2_set = set(s2.split())
    if not s1_set and not s2_set:
        warning('got 2 empty strings; making distance to be 1')
        return 1
    inter = s1_set.intersection(s2_set)
    dist = 1 - len(inter) / (len(s1_set) + len(s2_set) - len(inter))
    return dist


def get_exception_messages(workdir, truncate=True):
    e_msg_list = []
    e_fname_list = []
    for root, _, files in os.walk(workdir):
        # NOTE: we assume there's no subdir in the workdir
        for name in files:
            if not name.endswith('.e'):
                continue
            e_fname_list.append(name)
            e_path = os.path.join(root, name)
            with open(e_path, 'r') as rf:
                e_msg = rf.read()
            if truncate:
                e_msg = e_msg[:200]  # only take the first 200 char
            e_msg_list.append(e_msg)
    return e_msg_list, e_fname_list


def get_default_range(n_bits, np_dtype):
    if 'float' in np_dtype:
        low = np.finfo(np_dtype).min
        high = np.finfo(np_dtype).max
        # NOTE: if it's 64 bits, we have to half the low and high to have a representable range (high - low) w/ float64
        if n_bits == '64':
            low /= 2
            high /= 2
    elif 'int' in np_dtype:
        low = np.iinfo(np_dtype).min
        high = np.iinfo(np_dtype).max
    else:
        print('Error: cannot get range for dtype %s' % np_dtype)
        exit(1)
    return low, high


def choose_epsilon(np_dtype):
    if 'float' in np_dtype:
        return 1e-8
    elif 'int' in np_dtype:
        return 1
    print('Error: cannot get epsilon for dtype: %s' % np_dtype)


def DEBUG(msg):
    print('<<< DEBUG %s >>>' % msg)


def is_iterable(obj):
    return isinstance(obj, Iterable)
