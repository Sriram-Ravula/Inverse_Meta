cd

cd Inverse_Meta/

conda activate inverse_meta


python3 main_new.py --config configs/new_tests/ddrm_brain_hor.yml --doc TEST_ddrm_brain_hor --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_rand.yml --doc TEST_ddrm_brain_rand --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_vert.yml --doc TEST_ddrm_brain_vert --eta 0.8 --etaB 0.9 --timesteps 1000


#new tests here
python3 main_new.py --config configs/new_tests/ddrm_brain_hor_lowR.yml --doc TEST_ddrm_brain_hor_lowR --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_rand_highR.yml --doc TEST_ddrm_brain_rand_highR --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_rand_lowR.yml --doc TEST_ddrm_brain_rand_lowR --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_rand_soft_lowR.yml --doc TEST_ddrm_brain_rand_soft_lowR --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_rand_soft_highR.yml --doc TEST_ddrm_brain_rand_soft_highR --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/ddrm_brain_hor_soft_highR.yml --doc TEST_ddrm_brain_hor_soft_highR --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/ddrm_brain_hor_soft_lowR.yml --doc TEST_ddrm_brain_hor_soft_lowR --eta 0.8 --etaB 0.9 --timesteps 1000
#new tests here

#NOTE LATEST TESTS
python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Hard_Poisson_R4.yml --doc TEST_2D_Hard_Poisson_R4 --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Hard_Poisson_R10.yml --doc TEST_2D_Hard_Poisson_R10 --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Hard_Random_R4.yml --doc TEST_2D_Hard_Random_R4 --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Hard_Random_R10.yml --doc TEST_2D_Hard_Random_R10 --eta 0.8 --etaB 0.9 --timesteps 1000


python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Soft_Poisson_R4.yml --doc TEST_2D_Soft_Poisson_R4 --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Soft_Poisson_R10.yml --doc TEST_2D_Soft_Poisson_R10 --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Soft_Random_R4.yml --doc TEST_2D_Soft_Random_R4 --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config /home/sravula/Inverse_Meta/configs/new_tests/2D_Soft_Random_R10.yml --doc TEST_2D_Soft_Random_R10 --eta 0.8 --etaB 0.9 --timesteps 1000
#NOTE LATEST TESTS


python3 main_new.py --config configs/new_tests/ddrm_knee_hor.yml --doc TEST_ddrm_knee_hor --eta 0.7 --etaB 1.0 --timesteps 1000  

python3 main_new.py --config configs/new_tests/ddrm_knee_rand.yml --doc TEST_ddrm_knee_rand --eta 0.7 --etaB 1.0 --timesteps 1000  

python3 main_new.py --config configs/new_tests/ddrm_knee_vert.yml --doc TEST_ddrm_knee_vert --eta 0.7 --etaB 1.0 --timesteps 1000  




python3 main_new.py --config configs/new_tests/wavelet_brain_hor.yml --doc TEST_wavelet_brain_hor 

python3 main_new.py --config configs/new_tests/wavelet_brain_rand.yml --doc TEST_wavelet_brain_rand 

python3 main_new.py --config configs/new_tests/wavelet_brain_vert.yml --doc TEST_wavelet_brain_vert



python3 main_new.py --config configs/new_tests/wavelet_knee_hor.yml --doc TEST_wavelet_knee_hor 

python3 main_new.py --config configs/new_tests/wavelet_knee_rand.yml --doc TEST_wavelet_knee_rand 

python3 main_new.py --config configs/new_tests/wavelet_knee_vert.yml --doc TEST_wavelet_knee_vert