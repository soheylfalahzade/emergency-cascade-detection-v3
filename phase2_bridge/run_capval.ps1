$py = "C:\Users\Admin\carla_venv\Scripts\python.exe"
foreach ($w in 'ClearNoon','HardRainNoon','ClearNight','Fog') { foreach ($s in 10,11,12) {
  $lg = "capval_${w}_${s}.log"
  if ((Test-Path $lg) -and (Get-Content $lg | Select-String '^\{"source"')) { continue }
  Remove-Item "captures_val\${w}_s${s}" -Recurse -ErrorAction SilentlyContinue
  & $py -u carla_capture.py --weather $w --seed $s --duration 100 --warmup 15 --cam-height 16 --cam-pitch -28 --fov 70 --emerg-every 17 --emerg-dists 28,33,38 --inflow 0.35 --out captures_val > $lg 2>&1
}}
"done" | Set-Content done_capval.txt
