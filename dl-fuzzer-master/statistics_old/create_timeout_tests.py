import argparse
import os

timeoutCountDict = {}

def writeInputTest(out_home, api_name, pInput):
  #print (api_name)
  #print (pInput)
  out_api_dir = out_home + os.path.sep + api_name
  if not os.path.exists(out_api_dir):
    os.makedirs(out_api_dir)
  out_file_name = api_name + "." + "timeout" + str(timeoutCountDict[api_name]) + ".py"  #tf.compat.v1.bincount.timeout1.py
  out_path = out_api_dir + os.path.sep + out_file_name
#  s = "import tensorflow as tf\n"
#  s += "import pickle\n"
#  s += "data = pickle.load(open(\"" + pInput + "\",\'rb\'))\n"
#  s += api_name + "(**data)"
  #print (out_path)
  #print (s)
  
  with open(out_path, mode='w') as f:
      f.write("import tensorflow as tf\n")
      f.write("import pickle\n\n")
      f.write("data = pickle.load(open(\"" + pInput + "\",\'rb\'))\n")
      f.write("print(data)\n")
      f.write(api_name + "(**data)\n")

def updateDict(api_name):
  if api_name in timeoutCountDict:
    timeoutCountDict[api_name] += 1
  else:
    timeoutCountDict[api_name] = 1

def extractTestInfo(text, home_dir):
  #Given text: ./tf.compat.v1.bincount.yaml_workdir/out-Saving seed to file /home/workdir/expect_ok_prev_ok/tf.compat.v1.bincount.yaml_workdir/646c9b4754de4d22ebb0642f8fabdeb25f170513.p
  yaml = text.split("/")[1][:-len('.yaml_workdir')] #remove .yaml_workdir from tf.compat.v1.bincount.yaml_workdir
  pFilePath = text.rsplit(" ", 2)[1] # extract /home/workdir/expect_ok_prev_ok/tf.compat.v1.bincount.yaml_workdir/646c9b4754de4d22ebb0642f8fabdeb25f170513.p'\n'
  pFilePath = pFilePath.replace("/home", home_dir)
  return yaml, pFilePath 

def main(out_dir, log_file, home_dir):
  with open(log_file, 'r') as f:
    for line in f.readlines():
      if 'Saving seed to file' in line:
        api, pFile = extractTestInfo(line, home_dir)
        #print (api)
        #print (pFile)
        updateDict(api)
        writeInputTest(out_dir, api, pFile)

  print ("Test inputs have been wrttien for %d APIs." % len(timeoutCountDict))
  for key in timeoutCountDict.keys():
    print ("%s : %d" % (key, timeoutCountDict[key]))

if __name__== '__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--timeoutdir', help='Timeout directory that stores output test files')
  parser.add_argument('--input', help='Absolute path to grepped log file with keyword grep -r --include "out" "Timed Out" . -B 2')
  parser.add_argument('--mount', help='Absolute path of mounted folder to docker. e.g., /local1/m346kim/dl-fuzzing/docker_4_f19509')
  args = parser.parse_args()
  timeoutdir = args.timeoutdir
  log = args.input
  homedir = args.mount

  main(timeoutdir, log, homedir)
