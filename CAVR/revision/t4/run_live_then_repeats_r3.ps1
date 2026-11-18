# Third review round, unattended GPU chain on the 2x A10 node (launched detached via WMI Win32_Process Create).
# 1. Live pipeline (retrieval + router in the loop): front end on GPU 1 (this Windows process), generation on the vLLM
#    replica already running on GPU 0 in WSL; cavr_ce and always_visual at the original run's planned rates, seed 1234.
# 2. Then the repeat matrix in WSL (run_repeats_r3.sh repeats; it stops and restarts vLLM itself).
# Log: results/a10/live_pipeline_r3/chain.log
$ErrorActionPreference = "Continue"
$root = "D:\Cost-Aware-Visual-Router\CAVR\revision"
Set-Location $root
$log = "$root\results\a10\live_pipeline_r3\chain.log"
function Write-Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -FilePath $log -Append -Encoding utf8 }
$env:CUDA_VISIBLE_DEVICES = "1"
$py = "D:\envs\cavr\python.exe"
$runs = @(
  @{policy = "cavr_ce"; rates = "0.303,0.605,1.009,1.412,1.614,1.815,2.017,2.219"},
  @{policy = "always_visual"; rates = "0.35,0.7,1.166,1.632,1.865,2.099,2.332,2.565"}
)
foreach ($r in $runs) {
  Write-Log "start live pipeline $($r.policy) rates $($r.rates)"
  & $py live_pipeline_r3.py run --policy $r.policy --rates $r.rates *>> "$root\results\a10\live_pipeline_r3\run_$($r.policy).log"
  Write-Log "end live pipeline $($r.policy) exit $LASTEXITCODE"
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES
Write-Log "start repeats (WSL)"
& wsl.exe -d Ubuntu-24.04 -- bash /home/a10gpu/cavr_serve/run_repeats_r3.sh repeats
Write-Log "end repeats exit $LASTEXITCODE"
