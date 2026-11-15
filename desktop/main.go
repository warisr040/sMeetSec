// sMeetSec (simply handle IT) desktop — opens the app in a native Windows window (WebView2),
// the same way as sUML / sGenko. It starts the local Python backend (127.0.0.1 only),
// waits until it answers, shows the app, and stops the backend when the window closes.
// Nothing is sent to the internet.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/jchv/go-webview2"
)

const (
	appTitle = "sMeetSec 0.3 — simply handle IT"
	wmClose  = 0x0010
)

var (
	user32          = syscall.NewLazyDLL("user32.dll")
	shell32         = syscall.NewLazyDLL("shell32.dll")
	setWindowLongPt = user32.NewProc("SetWindowLongPtrW")
	callWindowProc  = user32.NewProc("CallWindowProcW")
	postMessage     = user32.NewProc("PostMessageW")
	messageBox      = user32.NewProc("MessageBoxW")
	shellExecute    = shell32.NewProc("ShellExecuteW")

	oldProc    uintptr
	closeMu    sync.Mutex
	allowClose bool
	closeAcked bool
)

func msg(text string) {
	t, _ := syscall.UTF16PtrFromString(text)
	c, _ := syscall.UTF16PtrFromString("sMeetSec")
	messageBox.Call(0, uintptr(unsafe.Pointer(t)), uintptr(unsafe.Pointer(c)), 0x40)
}

func openURL(u string) {
	verb, _ := syscall.UTF16PtrFromString("open")
	target, _ := syscall.UTF16PtrFromString(u)
	shellExecute.Call(0, uintptr(unsafe.Pointer(verb)), uintptr(unsafe.Pointer(target)), 0, 0, 1)
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// Fixed port keeps browser storage (theme) stable between runs; fall back to any free port.
func pickPort() int {
	for _, p := range []int{47393, 47394, 47395} {
		if l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p)); err == nil {
			l.Close()
			return p
		}
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 47393
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// The backend is, in order: a PyInstaller build (backend\sMeetSec-server.exe), the private Python made by
// the first-start setup (runtime\python), or a .venv made by hand.
func backendCommand(dir string, port int) (*exec.Cmd, error) {
	ps := fmt.Sprint(port)
	if p := filepath.Join(dir, "backend", "sMeetSec-server.exe"); exists(p) {
		return exec.Command(p, "--no-browser", "--port", ps), nil
	}
	if p := filepath.Join(dir, "runtime", "python", "pythonw.exe"); exists(p) && exists(filepath.Join(dir, "runtime", "ready.txt")) {
		return exec.Command(p, filepath.Join(dir, "run.py"), "--no-browser", "--port", ps), nil
	}
	if p := filepath.Join(dir, ".venv", "Scripts", "pythonw.exe"); exists(p) {
		return exec.Command(p, filepath.Join(dir, "run.py"), "--no-browser", "--port", ps), nil
	}
	return nil, fmt.Errorf("setup")
}

func hidden(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000} // CREATE_NO_WINDOW
}

func jsString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

const splash = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#e9ebf0;font-family:'Segoe UI',sans-serif;color:#1d2330}
.c{text-align:center}.s{width:38px;height:38px;border:4px solid #c9cdf7;border-top-color:#4f46e5;border-radius:50%;animation:r 1s linear infinite;margin:18px auto}
@keyframes r{to{transform:rotate(360deg)}}small{color:#6b7380}</style></head><body><div class="c">
<b style="font-size:20px">sMeetSec</b><div class="s"></div><small>Starting the local engine… (first start can take up to a minute)</small></div></body></html>`

const setupPage = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#e9ebf0;font-family:'Segoe UI',sans-serif;color:#1d2330}
.c{width:720px;background:#fff;border-radius:16px;padding:26px 30px;box-shadow:0 16px 50px rgba(30,27,75,.15)}
h2{margin:0 0 6px}p{color:#4a5160;line-height:1.55}
button{font:inherit;background:#4f46e5;color:#fff;border:0;border-radius:8px;padding:9px 20px;cursor:pointer;font-weight:600}
button:disabled{opacity:.5}
#step{font-weight:600;margin:14px 0 6px}
pre{background:#f4f5f8;border-radius:10px;padding:10px;height:220px;overflow:auto;font-size:11.5px;white-space:pre-wrap;display:none}
.bar{height:8px;background:#e6e8ef;border-radius:5px;overflow:hidden;margin:6px 0 10px;display:none}.bar i{display:block;height:100%;width:0;background:#4f46e5;transition:width .4s}
</style></head><body><div class="c">
<h2>Welcome to sMeetSec</h2>
<p>First start: sMeetSec needs to download its engine once - a private copy of Python, the speech-recognition packages and the
speech / speaker models (about 1.5-2.5 GB, 10-30 minutes). Nothing is installed system-wide; everything goes into this folder.
After this, sMeetSec works fully offline.</p>
<p>For translation and reports also install <b>Ollama</b> (ollama.com) and run <code>ollama pull qwen3.5:9b</code> (16 GB RAM) or <code>qwen3.5:4b</code>.</p>
<button id="go" onclick="this.disabled=true;smeetsecSetup()">Download and set up</button>
<div id="step"></div><div class="bar" id="bar"><i id="fill"></i></div><pre id="log"></pre>
</div><script>
function setStep(t,n){document.getElementById('step').textContent=t;document.getElementById('bar').style.display='block';document.getElementById('fill').style.width=(n*20)+'%'}
function addLog(t){const p=document.getElementById('log');p.style.display='block';p.textContent+=t+'\n';if(p.textContent.length>60000)p.textContent=p.textContent.slice(-40000);p.scrollTop=p.scrollHeight}
function failed(t){setStep('Setup failed: '+t,0);const b=document.getElementById('go');b.disabled=false;b.textContent='Try again'}
</script></body></html>`

// runSetup runs tools\setup.ps1 hidden and streams its output into the setup page.
func runSetup(w webview2.WebView, dir string, done func()) {
	logPath := filepath.Join(dir, "data", "setup.log")
	_ = os.MkdirAll(filepath.Join(dir, "data"), 0o755)
	logf, _ := os.Create(logPath)
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", filepath.Join(dir, "tools", "setup.ps1"))
	cmd.Dir = dir
	hidden(cmd)
	out, err := cmd.StdoutPipe()
	if err != nil {
		w.Dispatch(func() { w.Eval("failed(" + jsString(err.Error()) + ")") })
		return
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		w.Dispatch(func() { w.Eval("failed(" + jsString(err.Error()) + ")") })
		return
	}
	sc := bufio.NewScanner(out)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if logf != nil {
			logf.WriteString(line + "\r\n")
		}
		l := line
		if len(l) > 6 && l[:5] == "STEP " && l[6] == '/' {
			n := int(l[5] - '0')
			w.Dispatch(func() { w.Eval(fmt.Sprintf("setStep(%s,%d)", jsString(l), n)) })
		}
		w.Dispatch(func() { w.Eval("addLog(" + jsString(l) + ")") })
	}
	err = cmd.Wait()
	if logf != nil {
		logf.Close()
	}
	if err != nil || !exists(filepath.Join(dir, "runtime", "ready.txt")) {
		msgText := "see data\\setup.log"
		if err != nil {
			msgText = err.Error() + " - " + msgText
		}
		w.Dispatch(func() { w.Eval("failed(" + jsString(msgText) + ")") })
		return
	}
	done()
}

func main() {
	exe, _ := os.Executable()
	dir := filepath.Dir(exe)
	port := pickPort()
	url := fmt.Sprintf("http://127.0.0.1:%d/", port)
	_ = os.MkdirAll(filepath.Join(dir, "data"), 0o755)

	dataDir := filepath.Join(os.Getenv("LOCALAPPDATA"), "sMeetSec")
	_ = os.MkdirAll(dataDir, 0o755)
	w := webview2.NewWithOptions(webview2.WebViewOptions{
		Debug:     false,
		AutoFocus: true,
		DataPath:  dataDir,
		WindowOptions: webview2.WindowOptions{
			Title:  appTitle,
			Width:  1440,
			Height: 900,
			IconId: 2,
			Center: true,
		},
	})
	if w == nil {
		msg("sMeetSec needs the Microsoft Edge WebView2 Runtime.\n\nInstall it from:\nhttps://developer.microsoft.com/microsoft-edge/webview2/\n\nYou can also use sMeetSec in your browser: run setup_windows.bat, then run.bat")
		return
	}
	defer w.Destroy()
	hwnd := uintptr(w.Window())

	// ---- engine (local Python backend) ----
	var (
		engMu  sync.Mutex
		engine *exec.Cmd
	)
	stop := func() {
		engMu.Lock()
		defer engMu.Unlock()
		if engine != nil && engine.Process != nil {
			// kill the whole tree: the engine may have started llama-server.exe
			k := exec.Command("taskkill", "/PID", fmt.Sprint(engine.Process.Pid), "/T", "/F")
			hidden(k)
			if k.Run() != nil {
				_ = engine.Process.Kill()
			}
			engine = nil
		}
	}
	defer stop()

	startEngine := func() {
		cmd, err := backendCommand(dir, port)
		if err != nil {
			w.Dispatch(func() { w.SetHtml(setupPage) })
			return
		}
		cmd.Dir = dir
		hidden(cmd)
		logf, e := os.Create(filepath.Join(dir, "data", "server.log"))
		if e == nil {
			cmd.Stdout, cmd.Stderr = logf, logf
		}
		if err := cmd.Start(); err != nil {
			w.Dispatch(func() {
				w.SetHtml(`<body style="font-family:Segoe UI;padding:40px"><h3>Could not start the sMeetSec engine</h3><p>` + err.Error() + `</p></body>`)
			})
			return
		}
		engMu.Lock()
		engine = cmd
		engMu.Unlock()
		w.Dispatch(func() { w.SetHtml(splash) })

		exited := make(chan error, 1)
		go func() {
			exited <- cmd.Wait()
			if logf != nil {
				logf.Close()
			}
		}()
		client := http.Client{Timeout: 2 * time.Second}
		deadline := time.Now().Add(180 * time.Second)
		for time.Now().Before(deadline) {
			select {
			case <-exited:
				w.Dispatch(func() {
					w.SetHtml(`<body style="font-family:Segoe UI;padding:40px"><h3>The sMeetSec engine stopped.</h3>
<p>See <code>data\server.log</code> in the sMeetSec folder. Deleting the <code>runtime</code> folder and starting sMeetSec.exe again re-runs the setup.</p>
<button onclick="smeetsecLog()">Open server.log</button></body>`)
				})
				return
			default:
			}
			if r, e := client.Get(url + "api/guard"); e == nil {
				r.Body.Close()
				if r.StatusCode == 200 {
					w.Dispatch(func() { w.Navigate(url) })
					return
				}
			}
			time.Sleep(400 * time.Millisecond)
		}
		w.Dispatch(func() {
			w.SetHtml(`<body style="font-family:Segoe UI;padding:40px"><h3>The sMeetSec engine did not start in time.</h3><p>See data\server.log.</p></body>`)
		})
	}

	// ---- bindings ----
	var setupOnce sync.Mutex
	_ = w.Bind("smeetsecSetup", func() {
		go func() {
			if !setupOnce.TryLock() {
				return
			}
			defer setupOnce.Unlock()
			runSetup(w, dir, startEngine)
		}()
	})
	// Open the same app in the default browser (needed for "system audio" capture in live mode).
	_ = w.Bind("smeetsecOpenBrowser", func() { openURL(url) })
	_ = w.Bind("smeetsecOpenModels", func() {
		_ = os.MkdirAll(filepath.Join(dir, "models"), 0o755)
		openURL(filepath.Join(dir, "models"))
	})
	_ = w.Bind("smeetsecLog", func() { openURL(filepath.Join(dir, "data", "server.log")) })

	// Close button: let the page ask first when a live meeting or a job is running.
	_ = w.Bind("smeetsecCloseAck", func() {
		closeMu.Lock()
		closeAcked = true
		closeMu.Unlock()
	})
	_ = w.Bind("smeetsecQuit", func() {
		closeMu.Lock()
		allowClose = true
		closeMu.Unlock()
		postMessage.Call(hwnd, wmClose, 0, 0)
	})
	cb := syscall.NewCallback(func(h, m, wp, lp uintptr) uintptr {
		if m == wmClose {
			closeMu.Lock()
			ok := allowClose
			closeAcked = false
			closeMu.Unlock()
			if !ok {
				w.Eval("window.__smeetsecRequestClose ? window.__smeetsecRequestClose() : smeetsecQuit()")
				go func() { // safety net if the page does not answer
					time.Sleep(2 * time.Second)
					closeMu.Lock()
					acked := closeAcked
					if !acked {
						allowClose = true
					}
					closeMu.Unlock()
					if !acked {
						postMessage.Call(h, wmClose, 0, 0)
					}
				}()
				return 0
			}
		}
		r, _, _ := callWindowProc.Call(oldProc, h, m, wp, lp)
		return r
	})
	gwlpWndProc := ^uintptr(3) // GWLP_WNDPROC = -4
	oldProc, _, _ = setWindowLongPt.Call(hwnd, gwlpWndProc, cb)

	w.SetSize(1100, 700, webview2.HintMin)
	w.SetSize(1440, 900, webview2.HintNone)
	if _, err := backendCommand(dir, port); err != nil {
		w.SetHtml(setupPage) // first start: offer the one-time setup
	} else {
		w.SetHtml(splash)
		go startEngine()
	}

	w.Run()
	stop()
}
