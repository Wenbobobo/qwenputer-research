# CPU demo runner. Run only when free RAM >= 25 GB and no other heavy python jobs are running.
# Usage: pwsh -File C:\Projects\Qcompute\demo\run-demo.ps1 [-Step smoke|a|b|calc]
# NOTE: use pwsh 7 (not Windows PowerShell 5.1): 5.1 treats native stderr as a terminating error.
param([string]$Step = "smoke")
$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false
$wd = "C:\Projects\Qcompute\repro\qwenputer\experimental\riscv"
$py = "$wd\.venv\Scripts\python.exe"
$model = "C:\Projects\Qcompute\.data\Qwen3.5-4B-Qwenputer-Filesystem"
$demo = "C:\Projects\Qcompute\demo"
$freeGB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($freeGB -lt 25) { Write-Error ("free RAM {0:N1} GB < 25 GB; refusing to start" -f $freeGB); exit 2 }
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
switch ($Step) {
  "smoke" { $a = @("--ask","Hi","--max-new","32"); $log = "run-smoke.log" }
  "a"     { $a = @("--memory","$demo\store.npz","--ask","Remember that my locker code is 4471-AZ.","--max-new","120"); $log = "run-a.log" }
  "b"     { $a = @("--memory","$demo\store.npz","--ask","What's my locker code?","--max-new","120"); $log = "run-b.log" }
  "calc"  { $a = @("--ask","Use calc: what is 123456789 * 987654321?","--max-new","300"); $log = "run-calc.log" }
}
Push-Location $wd
$t0 = Get-Date
& $py -u -m chipformer.rv.chat $model @a *> "$demo\$log"
"exit=$LASTEXITCODE elapsed_s=$([int]((Get-Date)-$t0).TotalSeconds)" | Add-Content "$demo\$log"
Pop-Location
