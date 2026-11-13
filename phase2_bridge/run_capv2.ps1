$py = "C:\Users\Admin\carla_venv\Scripts\python.exe"
foreach ($w in 'ClearNoon','HardRainNoon','ClearNight','Fog') { foreach ($s in 0,1,2,3,4) {
  $lg = "capv2_${w}_${s}.log"
  if ((Test-Path $lg) -and (Get-Content $lg | Select-String '^\{"source"')) { continue }
  Remove-Item "captures_v2\${w}_s${s}" -Recurse -ErrorAction SilentlyContinue
  & $py -u carla_capture.py --weather $w --seed $s --duration 100 --warmup 15 --cam-height 15 --cam-pitch -25 --inflow 0.4 --out captures_v2 > $lg 2>&1
}}
"done" | Set-Content done_capv2.txt
