import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

#################################################
# author: Federico Menegazzo                    #
#                                               #
# contacts: federico.menegazzo2@studio.unibo.it #
#           federicomenegazzo01@gmail.com       #
#                                               #
# date: 17/12/2025                              #
#################################################





class signal:

    def __init__(self, n):

        self.__n = n
        self.__image = np.zeros( (n,) )
    

    def __getitem__(self, idx):
        return self.__image[idx]
    
    def __setitem__(self, idx, value):
        self.__image[idx] = value
    
    def __len__(self):
        return self.__n

    def __generate_interpolator(self, F, x_0, y_0, x_1, y_1):
        
        f = lambda x: ( y_1 * F( (x - x_0) / (x_1 - x_0) ) + y_0 * (1 - F( (x - x_0) / (x_1 - x_0) )) )

        return f

    def __add__(self, other):

        if not(self.__n == other.__n):

            print("Critical Error: signal sizes do not match! Returning empty signal")
            return signal(self.__n)
        
        else:

            img_1 = self.__image
            img_2 = other.__image

            sum_signal = signal(self.__n)
            sum_signal.__image = img_1 + img_2

            return sum_signal
        
    def __mul__(self, other):

        if not( (type(other) == float) or (type(other) == np.float64)):

            print("Critical Error: can't compute this operation! Returning empty signal")
            return signal(self.__n)
        
        else:

            prod_signal = signal(self.__n)
            prod_signal.__image = self.__image * np.float32(other)
            return prod_signal
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        negative_other = (-1.0) * other
        return self.__add__(negative_other)

    def __is_bound_intersected(self, y_0, teta_0, alfa_kurv, d=1):

        n_abs_inf = int( abs(teta_0 - np.pi/2) // alfa_kurv )
        n_abs_sup = n_abs_inf + 1


        if (teta_0 < np.pi/2):

            y_max = y_0 + sum([ d * (np.cos(teta_0 + k * alfa_kurv)/np.sin(teta_0 + k * alfa_kurv)) for k in range(0, n_abs_sup)])

            if (y_max <= 1): return False
            else: return True
        
        if (teta_0 > np.pi/2):            

            y_min = y_0 + sum([ d * (np.cos(teta_0 - k * alfa_kurv)/np.sin(teta_0 - k * alfa_kurv)) for k in range(0, n_abs_sup)])

            if (y_min >= -1): return False
            else: return True
    

    def __get_compatible_alfa(self, y_0, teta_0, alfa_kurv, must_be_convessa, tol_rel = 1E-3):

        invert_convexity = False
        found_comp_alfa = False

        supp_comp = [-1 * alfa_kurv, alfa_kurv]
        

        #check zero's avaibility

        alfa_guess = 0
        teta_1 = teta_0 - alfa_guess

        alfa_is_valid = not( self.__is_bound_intersected(y_0, teta_1, alfa_kurv) )
        it_is_convex = True if (teta_1 > np.pi/2) else False

        if not(alfa_is_valid):

            if it_is_convex: the_assumption_can_be_met = must_be_convessa
            else: the_assumption_can_be_met = not(must_be_convessa)
        
        else:
            the_assumption_can_be_met = True

        
        if not(the_assumption_can_be_met):
            invert_convexity = True
        
        elif the_assumption_can_be_met and must_be_convessa:
            supp_comp[0] = 0
        
        elif the_assumption_can_be_met and not(must_be_convessa):
            supp_comp[1] = 0


        while (not(found_comp_alfa)):

            alfa_guess = np.random.uniform(max(supp_comp[0], -1 * (np.pi - teta_0) ), min(supp_comp[1], teta_0))

            teta_1 = teta_0 - alfa_guess

            alfa_is_valid = not( self.__is_bound_intersected(y_0, teta_1, alfa_kurv) )       

            it_is_convex = True if (teta_1 > np.pi/2) else False

            
            if not(alfa_is_valid) and ( (supp_comp[1] - supp_comp[0])/alfa_kurv >= tol_rel ) :

                if it_is_convex: supp_comp[0] = alfa_guess
                else: supp_comp[1] = alfa_guess
            
            elif not(alfa_is_valid) and ( (supp_comp[1] - supp_comp[0])/alfa_kurv < tol_rel ):

                alfa_guess = alfa_kurv * np.sign(supp_comp[0])
                found_comp_alfa = True                

            else:

                found_comp_alfa = True

        return alfa_guess, invert_convexity


    def __generate_partial_signal(self, l, start, teta_0, alfa_kurv, starting_convex):

        K = np.tan(alfa_kurv/2) * 2
        r = 1 / K
        max_convexity_len = int( np.sqrt(2 * r - 1) ) 
        
        if (l == 0): return np.array([])

        part_sign = np.zeros((l,))
        part_sign[0] = start


        convexity = starting_convex
        convex_pointer = np.random.randint(max_convexity_len)
        teta_k = teta_0
        y_k = start



        for k in range(1, l):

            [alfa_k, invert_convexity] = self.__get_compatible_alfa(y_0=y_k, teta_0=teta_k, alfa_kurv=alfa_kurv, must_be_convessa=convexity)
            convexity = not(convexity) if invert_convexity else convexity

            teta_k -= alfa_k
            y_k += np.cos(teta_k)/np.sin(teta_k)            

            part_sign[k] = y_k           

            convex_pointer -= 1

            if (convex_pointer <= 0):
                convexity = not(convexity)
                convex_pointer = np.random.uniform(max_convexity_len)
        

        return part_sign

    def get_image(self):
        return (self.__image).copy()

    def return_curvature_vect(self):

        kurvature_vect = np.zeros( (self.__n - 2,) )


        for k in range(0, self.__n - 2):

            y_0 = self.__image[k]
            y_1 = self.__image[k + 1]
            y_2 = self.__image[k + 2]

            v = np.array([-1, y_0 - y_1])
            w = np.array([1, y_2 - y_1])

            teta = np.acos( np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)) )

            kurvature_vect[k] = 2/np.tan(teta/2)
        

        return kurvature_vect

    def generate_GG_realization(self, media, std_dev, beta):

        N = self.__n

        # compute the GG pdf scale parameter alpha
        alpha = std_dev * np.sqrt( scipy.special.gamma(1/beta)/scipy.special.gamma(3/beta) )

        # generate the realization
        x = media + alpha * ( (np.random.gamma(1/beta,1,N)**(1/beta)) * ((np.random.uniform(0,1,size = N)<0.5)*2-1) )
        
        self.__image = x

    def generate_smooth_sign(self, K, must_be_normalized=True):

        K = max( 1/( (self.__n/4)**2 + 1 ), K )
        K = min(2 / self.__n, K)

        alfa_kurv = 2 * np.arctan(K/2)
        r = 1/K

        min_boundary_dist = min(math.ceil( 2 * np.sqrt(r - 1) ), self.__n//2)

        rnd_estr_idx = np.random.randint(min_boundary_dist, (self.__n - min_boundary_dist) + 1)
        

            
        convessità = False
        self.__image[rnd_estr_idx] = 1

        teta_sx = np.random.uniform( max(np.pi/2 - alfa_kurv, np.arctan(0.5) ), np.pi/2 )
        teta_dx = teta_sx - ( self.__get_compatible_alfa(y_0=1, teta_0=teta_sx, alfa_kurv=alfa_kurv, must_be_convessa=False) )[0]


        y_0_sx = -1 + np.cos(teta_sx)/np.sin(teta_sx)
        y_0_dx = 1 + np.cos(teta_dx)/np.sin(teta_dx)


        sign_sx = -1 * ( (self.__generate_partial_signal(l=rnd_estr_idx, start=y_0_sx, teta_0=teta_sx, alfa_kurv=alfa_kurv, starting_convex=not(convessità)))[::-1] )
        sign_dx = self.__generate_partial_signal(l=self.__n-1-rnd_estr_idx, start=y_0_dx, teta_0=teta_dx, alfa_kurv=alfa_kurv, starting_convex=convessità)


        self.__image[:rnd_estr_idx] = sign_sx
        self.__image[rnd_estr_idx + 1:] = sign_dx


        #normalization step

        if (must_be_normalized):

            imm_0 = self.__image[0]

            r_sx = (rnd_estr_idx**2 + (1 + imm_0)**2) / ( 2 * (1 + imm_0) )
            r_dx = ( (self.__n - rnd_estr_idx) / 2)**2 + 1

            C_sx = lambda x: np.sqrt(r_sx**2 - (x - rnd_estr_idx)**2) - (r_sx - 1)
            C_dx = lambda x: np.sqrt(r_dx**2 - (x - rnd_estr_idx)**2) - (r_dx - 1)

            for j in range(self.__n):

                if (j <= rnd_estr_idx):

                    self.__image[j] = (self.__image[j] + C_sx(j))/2
                
                else:

                    self.__image[j] = ( self.__image[j] + C_dx(j))/2
                
            

        if (np.random.randint(0,2) == 1):
            self.__image = -1 * self.__image




    def generate_oscillatory_sign(self, interpolating_function, n_ext, K, T_min=10, std_sin=False):

        smooth_up = signal(self.__n)
        smooth_down = signal(self.__n)

        if n_ext == 1:

            T_min = self.__n
            T_max = self.__n
        
        elif n_ext > self.__n:

            T_min = 1
            T_max = 1
            n_ext = self.__n
        
        else:
            
            T_min = int( np.floor( min(self.__n / n_ext, T_min) ) )
            T_max = T_min + int( np.floor( (self.__n - T_min * (n_ext)) / (n_ext - 1) ) )
        
        
        

        smooth_up.__image = np.ones( (self.__n, ) )
        smooth_down = -1.0 * smooth_up
        imm_0 = 0

        
        c_0 = 0
        up = False
        ext_done = 0


        while (ext_done != n_ext):

            c_1 = min( c_0 + np.random.randint(T_min, T_max + 1), self.__n)

            if up:
                imm_1 = smooth_up[ min(c_1,self.__n - 1) ]

            else:
                imm_1 = smooth_down[ min(c_1,self.__n - 1) ]

            w_0_1 = self.__generate_interpolator(interpolating_function, c_0, imm_0, c_1, imm_1)

            self.__image[c_0:c_1] = w_0_1( np.arange(c_0, c_1) )


            up = not(up)
            c_0 = c_1       
            imm_0 = imm_1

            ext_done += 1

        
        while (c_0 != self.__n):

            c_1 = min( c_0 + np.random.randint(T_min, T_max + 1), self.__n)

            if up:
                imm_1 = smooth_up[ min(c_1,self.__n - 1) ]

            else:
                imm_1 = smooth_down[ min(c_1,self.__n - 1) ]

            w_0_1 = self.__generate_interpolator(interpolating_function, c_0, imm_0, c_1, imm_1)

            self.__image[c_0:c_1] = w_0_1( np.arange(c_0, c_1) )


            up = not(up)
            c_0 = c_1       
            imm_0 = imm_1

        
        if not(std_sin):

            smooth_modulus = signal(self.__n)            
            smooth_modulus.generate_smooth_sign(K, must_be_normalized=False)

            self.__image = self.__image * smooth_modulus.get_image()
        

        #normalization step

        self.__image = self.__image * (1/max(np.abs(self.__image)))

        if not(std_sin):
            return smooth_modulus          
        
    def generate_cartoon_sign(self, n_partitions, T_min, p=0.95):

        n_jumps = int( np.ceil( (np.log(n_partitions) - np.log(1-p)) / (np.log(n_partitions) - np.log(n_partitions - 1))) ) if (n_partitions > 1) else 1

        if n_jumps == 1:

            n_jumps += 1  

        if n_jumps > self.__n:

            T_min = 1
            T_max = 1
            n_jumps = self.__n
        
        else:
            
            T_min = int( np.floor( min(self.__n / n_jumps, T_min) ) )
            T_max = T_min + int( np.floor( (self.__n - T_min * (n_jumps)) / (n_jumps - 1) ) )

        current_idx = 0
        next_idx = 0
        jumps_done = 0


        while (jumps_done != n_jumps):

            next_idx = min( current_idx + np.random.randint(T_min, T_max + 1), self.__n)

            v = np.random.uniform(-1, 1)

            self.__image[current_idx : next_idx] = np.ones((next_idx - current_idx,)) * v

            current_idx = next_idx
            jumps_done += 1

        
        while (current_idx != self.__n):

            next_idx = min( current_idx + np.random.randint(T_min, T_max + 1), self.__n)

            v = np.random.uniform(-1, 1)

            self.__image[current_idx : next_idx] = np.ones((next_idx - current_idx,)) * v

            current_idx = next_idx
        

        self.__image = self.__image - np.ones(self.__n) * self.__image[0]
        self.__image = self.__image * (1/max(np.abs(self.__image)))
    
    def generate_blended_signal(self, snr_tuple, beta_tuple, T_min_cartoon, prob_std_sin, kernel='cos'):

        s = signal(self.__n)
        o = signal(self.__n)
        c = signal(self.__n)
        n = signal(self.__n)

        if (kernel == 'cos'):

            interp = lambda x: (-np.cos(x * np.pi) + 1)/2   

        elif (kernel == 'lin'):

            interp = lambda x: x 

        else:

            interp = lambda x: np.heaviside( (x - 1), 1 )


        snr = np.random.uniform(snr_tuple[0], snr_tuple[1])
        beta = np.random.uniform(beta_tuple[0], beta_tuple[1])

        K_smooth = np.random.uniform( 1/( (self.__n/4)**2 + 1 ), 2/self.__n )
        K_osc = np.random.uniform( 1/( (self.__n/4)**2 + 1 ), 2/self.__n )
        n_ext = np.random.randint(int(self.__n/20), int(self.__n/10) )
        n_partitions = np.random.uniform(1, 10)

        std_sin = True if (np.random.uniform() <= prob_std_sin) else False

        s.generate_smooth_sign(K=K_smooth)
        o.generate_oscillatory_sign(interp, n_ext=n_ext, K=K_osc, std_sin=std_sin)
        c.generate_cartoon_sign(n_partitions=n_partitions, T_min=T_min_cartoon)

        beta_s = np.random.uniform(0, 1)
        beta_o = np.random.uniform(0, 1 - beta_s)
        beta_c = 1 - beta_s - beta_o

        f = beta_s * s + beta_o * o + beta_c * c

        norma_segnale = np.linalg.norm(f.__image)
        dev_std_noise = norma_segnale / ( np.sqrt(10**(snr/10) * self.__n) )

        n.generate_GG_realization(0, dev_std_noise, beta)
        n[0] = 0 

        self.__image = f.__image + n.__image

        inf_norm = max(np.abs(self.__image))

        self.__image *= (1/inf_norm)

        s *= (beta_s / inf_norm)
        o *= (beta_o / inf_norm)
        c *= (beta_c / inf_norm)
        n *= (1 / inf_norm)
   

        return (s, o, c, n)







            







            

        
