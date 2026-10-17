//! Bot processes, like `Bot` in halite.py: `/bin/sh -c COMMAND`, stdout read
//! line by line on a thread, stderr to a log file.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

pub struct Bot {
    pub tag: usize,
    pub command: String,
    pub name: String,
    pub no_timeout: bool,
    pub alive: bool,
    pub error: Option<String>,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    lines: Receiver<Option<String>>,
}

impl Bot {
    pub fn new(tag: usize, command: &str, log_path: Option<&str>, no_timeout: bool) -> Bot {
        let stderr = match log_path {
            Some(p) => File::create(p).map(Stdio::from).unwrap_or_else(|_| Stdio::null()),
            None => Stdio::null(),
        };
        let spawned = Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn();
        let (tx, rx) = channel();
        let (child, stdin) = match spawned {
            Ok(mut child) => {
                let stdin = child.stdin.take();
                let stdout = child.stdout.take().unwrap();
                thread::spawn(move || {
                    read_lines(stdout, |line| tx.send(Some(line)).is_ok());
                    let _ = tx.send(None);
                });
                (Some(child), stdin)
            }
            Err(_) => {
                let _ = tx.send(None);
                (None, None)
            }
        };
        Bot {
            tag,
            command: command.to_string(),
            name: command.to_string(),
            no_timeout,
            alive: true,
            error: None,
            child,
            stdin,
            lines: rx,
        }
    }

    pub fn send(&mut self, text: &str) {
        let ok = match self.stdin.as_mut() {
            Some(s) => {
                let mut buf = Vec::with_capacity(text.len() + 1);
                buf.extend_from_slice(text.as_bytes());
                buf.push(b'\n');
                s.write_all(&buf).and_then(|_| s.flush()).is_ok()
            }
            None => false,
        };
        if !ok {
            self.fail("crashed (pipe closed)");
        }
    }

    /// One line from the bot, or None after marking it failed.
    pub fn recv(&mut self, deadline: Instant) -> Option<String> {
        let got = if self.no_timeout {
            self.lines.recv().unwrap_or(None)
        } else {
            let wait = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(wait) {
                Ok(v) => v,
                Err(RecvTimeoutError::Timeout) => {
                    self.fail("timed out");
                    return None;
                }
                Err(RecvTimeoutError::Disconnected) => None,
            }
        };
        if got.is_none() {
            self.fail("crashed (exited)");
        }
        got
    }

    pub fn fail(&mut self, reason: &str) {
        if self.alive {
            self.alive = false;
            self.error = Some(reason.to_string());
        }
    }

    pub fn kill(&mut self) {
        if let Some(c) = self.child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
        self.stdin = None;
    }
}

/// Split a byte stream into lines the way Python's text-mode pipe does:
/// universal newlines ("\n", "\r\n" and a lone "\r" all end a line), UTF-8
/// decoding. A decoding error ends the stream, as the exception does in
/// halite.py's reader thread. A "\r" is only reported once the next byte (or
/// end of stream) shows whether a "\n" follows, as in Python.
fn read_lines<R: Read>(stream: R, mut emit: impl FnMut(String) -> bool) {
    let mut reader = BufReader::with_capacity(1 << 16, stream);
    let mut line: Vec<u8> = Vec::new();
    let mut buf = [0u8; 1 << 16];
    let mut pending_cr = false;
    let finish = |line: &mut Vec<u8>, emit: &mut dyn FnMut(String) -> bool| -> bool {
        let bytes = std::mem::take(line);
        match String::from_utf8(bytes) {
            Ok(s) => emit(s),
            Err(_) => false,
        }
    };
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &buf[..n] {
            if pending_cr {
                pending_cr = false;
                if !finish(&mut line, &mut emit) {
                    return;
                }
                if b == b'\n' {
                    continue;
                }
            }
            match b {
                b'\n' => {
                    if !finish(&mut line, &mut emit) {
                        return;
                    }
                }
                b'\r' => pending_cr = true,
                _ => line.push(b),
            }
        }
    }
    if pending_cr || !line.is_empty() {
        finish(&mut line, &mut emit);
    }
}

pub fn turn_deadline(timeout: f64) -> Instant {
    Instant::now() + Duration::from_secs_f64(timeout)
}
