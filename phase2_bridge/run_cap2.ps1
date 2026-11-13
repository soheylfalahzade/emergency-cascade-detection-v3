$py = "C:\Users\Admin\carla_venv\Scripts\python.exe"
foreach ($w in 'ClearNoon','HardRainNoon','ClearNight','Fog') { foreach ($s in 0,1,2) {
  $lg = "cap_${w}_${s}.log"
  if ((Test-Path $lg) -and (Get-Content $lg | Select-String '^\{"source"')) { continue }
  Remove-Item "captures_all\${w}_s${s}" -Recurse -ErrorAction SilentlyContinue
  & $py -u carla_capture.py --weather $w --seed $s --duration 100 --warmup 15 --cam-height 15 --cam-pitch -25 --emerg-every 20 --inflow 0.4 --out captures_all > $lg 2>&1
}}
"done" | Set-Content done_cap.txt
