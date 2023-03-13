import argparse
import concurrent.futures
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--script', default=[], nargs='+')
parser.add_argument('--argz', default=[], nargs='+')
opt = parser.parse_args()


def run_script(script, argz):
    subprocess.run(["python", script] + argz.split())


def run_scripts_parallel():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_script, script[0], script[1]) for script in zip(opt.script, opt.argz)]

        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == '__main__':
    run_scripts_parallel()
