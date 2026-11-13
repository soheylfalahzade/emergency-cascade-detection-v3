Set-Location C:\Users\Admin\Documents\phase2_bridge
$py = 'C:\Users\Admin\carla_venv\Scripts\python.exe'
$LOG = 'v11\driver.log'
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
function NoiseArgs($p) { return @('--p-miss',"$($p.p_miss_car)",'--p-miss-emerg',"$($p.p_miss_emerg)",'--p-miss-far',"$($p.p_miss_far)",'--far-m',"$($p.far_m)",'--p-emerg-as-car',"$($p.p_emerg_as_car)",'--fp-per-min',"$($p.fp_per_min)") }
function Have($out) {
  $h = @{}
  if (Test-Path $out) { Get-Content $out | ForEach-Object { if ($_.Trim()) { $j = $_ | ConvertFrom-Json; $h["$($j.policy)/$($j.seed)"] = 1 } } }
  return $h
}
function Run-One($pol, $s, $w, $extra, $out, $tag) {
  for ($a = 1; $a -le 2; $a++) {
    if (-not (Test-Carla)) { Start-Carla "server down before $tag" } elseif ($script:n -ge 10) { Start-Carla "preventive before $tag" }
    Remove-Item tmp_run.jsonl -ErrorAction SilentlyContinue
    $ar = @('-u','carla_runner.py','--policies',$pol,'--seeds',"$s",'--inflow','0.25','--n-traffic','20','--all-red','2')
    if ($w) { $ar += @('--weather',$w) }
    $ar += $extra
    & $py @ar --out tmp_run.jsonl > "v11\log_$tag.log" 2>&1
    $script:n++
    if (Test-Path tmp_run.jsonl) {
      $l = Get-Content tmp_run.jsonl | Select-Object -Last 1
      $ok = $true
      try { $null = $l | ConvertFrom-Json } catch { $ok = $false }
      if ($ok -and $l) { Add-Content $out $l; return }
    }
    Log "FAILED attempt $a $tag"
    Start-Carla "retry $tag"
  }
  Log "GIVEUP $tag"
}

Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'carla_runner|carla_capture' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Log "start v11"
& $py patch_adapter.py *> v11\patch.txt
if ($LASTEXITCODE -ne 0) { "patch failed" | Set-Content FAILED_v11.txt; exit 1 }
Log (Get-Content v11\patch.txt -Raw)
Start-Carla "initial"

# ---- gate: determinism with the patched adapter ----
Remove-Item v11\gate.jsonl -ErrorAction SilentlyContinue
$pf = (Get-Content noise_plan_ft.json -Raw | ConvertFrom-Json).ClearNoon
foreach ($pol in 'preempt','fuzzy') { foreach ($i in 1..3) { Run-One $pol 0 'ClearNoon' (NoiseArgs $pf) 'v11\gate.jsonl' "gate_${pol}_$i" } }
& $py gate_check.py *> v11\gate_check.txt
if ($LASTEXITCODE -ne 0) { "determinism gate failed, sweeps not run; see v11\gate_check.txt" | Set-Content FAILED_v11.txt; Log "GATE FAILED"; exit 1 }
Log "gate passed"

# ---- sweeps with measured noise (ft, p1) ----
foreach ($c in @(@{tag='ft'; plan='noise_plan_ft.json'; seeds=0..9}, @{tag='p1'; plan='noise_plan_phase1.json'; seeds=0..4})) {
  $plan = Get-Content $c.plan -Raw | ConvertFrom-Json
  foreach ($w in 'ClearNight','ClearNoon','Fog','HardRainNoon') {
    $out = "v11\results_v10_$($c.tag)_$w.jsonl"
    $have = Have $out
    foreach ($s in $c.seeds) { foreach ($pol in 'preempt','fuzzy') {
      if ($have.ContainsKey("$pol/$s")) { continue }
      Run-One $pol $s $w (NoiseArgs $plan.$w) $out "$($c.tag)_${w}_${pol}_$s"
    } }
  }
}
Log "ft/p1 done"

# ---- v10b: fuzzy with visibility proxy (uncalibrated) ----
$plan = Get-Content noise_plan_phase1.json -Raw | ConvertFrom-Json
foreach ($w in 'ClearNight','ClearNoon','Fog','HardRainNoon') {
  $out = "v11\results_v10b_p1_$w.jsonl"
  $have = Have $out
  foreach ($s in 0..4) {
    if ($have.ContainsKey("fuzzy/$s")) { continue }
    Run-One 'fuzzy' $s $w ((NoiseArgs $plan.$w) + @('--use-visibility')) $out "v10b_${w}_$s"
  }
}
Log "v10b done"

# ---- swept (injected, not measured) noise, no weather ----
foreach ($lv in @(@{n='mild'; pm='0.1'; pe='0.1'; fp='0.5'}, @{n='severe'; pm='0.3'; pe='0.3'; fp='2.0'})) {
  $out = "v11\results_lowflow_noise_$($lv.n).jsonl"
  $have = Have $out
  foreach ($s in 0..9) { foreach ($pol in 'preempt','fuzzy') {
    if ($have.ContainsKey("$pol/$s")) { continue }
    Run-One $pol $s '' @('--p-miss',$lv.pm,'--p-emerg-as-car',$lv.pe,'--fp-per-min',$lv.fp,'--hold-s','2') $out "swept_$($lv.n)_${pol}_$s"
  } }
}
Log "swept done"

# ---- tables ----
& $py table_v10.py v11 --csv v11\table_v11.csv 2>&1 | Out-File v11\table_v11.txt -Encoding UTF8
foreach ($lv in 'mild','severe') { "== swept $lv" | Add-Content v11\swept_summaries.txt; & $py summarize_carla.py "v11\results_lowflow_noise_$lv.jsonl" --ref preempt 2>&1 | Add-Content v11\swept_summaries.txt }
Log "tables done"

# ---- reproducibility audit: rerun stored rows ----
Remove-Item v11\audit_*.jsonl -ErrorAction SilentlyContinue
$pf = Get-Content noise_plan_ft.json -Raw | ConvertFrom-Json
$pp = Get-Content noise_plan_phase1.json -Raw | ConvertFrom-Json
foreach ($a in @(@('ft','ClearNoon',0), @('ft','Fog',3), @('p1','HardRainNoon',2), @('p1','ClearNight',1))) {
  $plan = if ($a[0] -eq 'ft') { $pf } else { $pp }
  foreach ($pol in 'preempt','fuzzy') {
    Run-One $pol $a[2] $a[1] (NoiseArgs $plan.($a[1])) "v11\audit_$($a[0])_$($a[1])_${pol}_$($a[2]).jsonl" "audit_$($a[0])_$($a[1])_${pol}_$($a[2])"
  }
}
& $py audit_check.py *> v11\audit_report.txt
Log "audit done"

# ---- backup + STATUS ----
$bk = 'results_phase2_carla_2026-11-10'
$null = New-Item -ItemType Directory -Force "$bk\prefix_unsorted_noise_results"
Copy-Item results_*.jsonl "$bk\prefix_unsorted_noise_results" -Force
Copy-Item v11 "$bk\v11" -Recurse -Force
Copy-Item *.py,*.ps1,noise_plan_*.json,STATUS.md,carla_adapter.py.bak_prefix,table_v10.csv,table_v10.txt $bk -Force -ErrorAction SilentlyContinue
$st = @('', '## 2026-11-10', 'Found: PerceptionNoise drew RNG values in the order of world.get_actors(), which is not stable across runs, so noisy runs (fuzzy and preempt) were not reproducible (e.g. stored p1 ClearNoon fuzzy seed 0: 49 s vs 2 s on rerun). Fix: carla_adapter.read_obs sorts vehicles by (lane, dist_m rounded to 0.01, cls) before noise.step (backup carla_adapter.py.bak_prefix).',
  'All noisy sweeps re-run into v11\ (ft, p1, v10b, swept mild/severe). Pre-fix noisy results are kept under prefix_unsorted_noise_results and are NOT reproducible; do not cite them. See v11\gate_check.txt, v11\audit_report.txt, v11\table_v11.txt.')
$st | Add-Content STATUS.md -Encoding UTF8
"done" | Set-Content done_v11.txt
Log "ALL DONE"
