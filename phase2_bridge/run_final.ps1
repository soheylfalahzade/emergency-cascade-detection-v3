Set-Location C:\Users\Admin\Documents\phase2_bridge
$py = 'C:\Users\Admin\carla_venv\Scripts\python.exe'
$LOG = 'heldout\driver.log'
$script:n = 0
function Log($m) { ("{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) | Add-Content $LOG }
function Test-Carla { $null = & $py -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(20);c.get_world().get_map().name" 2>$null; return ($LASTEXITCODE -eq 0) }
function Start-Carla($why) {
  Log "restart CARLA: $why"
  Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep 5
  Start-Process E:\CarlaUE4\CarlaUE4.exe -ArgumentList '-windowed','-ResX=1280','-ResY=720'
  $t = 0
  while (-not (Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue) -and $t -lt 120) { Start-Sleep 5; $t++ }
  Start-Sleep 30
  $script:n = 0
}
function Cap-One($w, $s) {
  $dir = "captures_heldout\${w}_s$s"
  for ($a = 1; $a -le 2; $a++) {
    if (-not (Test-Carla)) { Start-Carla "server down before $w $s" } elseif ($script:n -ge 5) { Start-Carla "preventive before $w $s" }
    Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
    & $py -u carla_capture.py --weather $w --seed $s --cam-height 13 --cam-pitch -22 --fov 80 --emerg-every 19 --emerg-dists 26,31,36 --inflow 0.30 --signal fixed --green 20 --yellow 3 --allred 2 --out captures_heldout > "heldout\log_cap_${w}_$s.log" 2>&1
    $script:n++
    $g = "$dir\gt.jsonl"
    if ((Test-Path $g) -and ((Get-Content $g | Measure-Object).Count -ge 60)) { Log "captured $w $s"; return }
    Log "FAILED attempt $a capture $w $s"
    Start-Carla "retry capture $w $s"
  }
  Log "GIVEUP capture $w $s"
}

Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'carla_runner|carla_capture' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Log "start final"
Start-Carla "initial"
foreach ($w in 'ClearNoon','ClearNight','Fog','HardRainNoon') { foreach ($s in 20..22) {
  $g = "captures_heldout\${w}_s$s\gt.jsonl"
  if ((Test-Path $g) -and ((Get-Content $g | Measure-Object).Count -ge 60)) { Log "have $w $s"; continue }
  Cap-One $w $s
} }
Log "captures done"

if (-not (Test-Carla)) { Start-Carla "before snapshot" }
& $py snap_junction.py > heldout\snap.log 2>&1
Log ("snapshot exit " + $LASTEXITCODE)

Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 10
& $py mk_figs_heldout.py > heldout\mkfig.log 2>&1
wsl bash /mnt/c/Users/Admin/Documents/phase2_bridge/run_final_eval.sh > heldout\eval.log 2>&1
Log ("heldout eval + figures done, exit " + $LASTEXITCODE)

& $py make_readme.py > heldout\readme.log 2>&1
Log ("README build exit " + $LASTEXITCODE)

$bk = 'results_phase2_carla_2026-11-10'
$null = New-Item -ItemType Directory -Force $bk
foreach ($d in 'carla_eval_heldout','figs_heldout','figs_sim','docs','heldout') { if (Test-Path $d) { Copy-Item $d "$bk\$d" -Recurse -Force } }
Copy-Item README.md,make_readme.py,make_sim_images.py,snap_junction.py,sync_repo.py,run_final.ps1,run_final_eval.sh -Destination $bk -Force -ErrorAction SilentlyContinue
@('', '## 2026-11-10 (final)', 'Held-out capture (seeds 20-22, camera 13 m / -22 / FOV 80) evaluated for carla_ft and yolov8n_fold0; junction phase snapshots; README built from result files; repo synced and committed (not pushed). Visual check of light-to-approach mapping: pending review of figs_sim\fig_sim_junction_phases.png.') | Add-Content STATUS.md -Encoding UTF8

wsl bash /mnt/c/Users/Admin/Documents/phase2_bridge/run_final_sync.sh > heldout\sync.log 2>&1
Log ("sync exit " + $LASTEXITCODE)
"done" | Set-Content done_final.txt
Log "ALL DONE"
