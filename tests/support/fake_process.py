"""Stand-in for a worker child process whose output the test scripts."""

import io
import subprocess


class FakeProcess:
    """Replaces subprocess.Popen for one child that prints output, then exits.

    The output is given as bytes and decoded as Popen's text mode decodes it.
    Like the operating system, it refuses an environment that is not all
    strings. A child that ignores termination only stops once it is killed.
    """

    def __init__(self, output=b"", exit_code=0, ignores_termination=False):
        self.output = output
        self.exit_code = exit_code
        self.ignores_termination = ignores_termination
        self.commands = []
        self.environments = []
        self.terminated = False
        self.killed = False
        self.waits = []

    def __call__(
        self, command, *, env, stdout, stderr, text=None, encoding=None, errors=None
    ):
        if env is not None and not all(
            isinstance(item, str) for pair in env.items() for item in pair
        ):
            raise TypeError("The environment must hold only strings.")
        self.commands.append(command)
        self.environments.append(env)
        raw = io.BytesIO(self.output)
        if encoding or errors or text:
            default = subprocess._text_encoding()
            self.stdout = io.TextIOWrapper(raw, encoding or default, errors)
        else:
            self.stdout = raw
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.stdout.close()
        return False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.ignores_termination and self.terminated and not self.killed:
            if timeout is None:
                raise RuntimeError("Waiting without a timeout would block forever.")
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.exit_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
