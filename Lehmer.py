import random
import numpy as np
class Lehmer():
    
    def __init__(self,lower_number,upper_number,exp,c):
        self.prime_numbers = []
        self.lower_number = lower_number
        self.upper_number = upper_number
        self.exp = exp
        self.m = 2**exp
        self.prime_number()
        self.a = random.choice(self.prime_numbers)
        self.x0 = random.choice(self.prime_numbers)
        self.c = c
        while self.a == self.x0:
            self.x0 = random.choice(self.prime_numbers)
            
    
    def prime_number(self):
        prime_numbers = []
        for num in range(self.lower_number, self.upper_number + 1):
            for i in range(2, int(num**(1/2))+1):
                if (num % i) == 0:
                    break
            else:
                self.prime_numbers.append(num)
    

    def generate_key(self):
        arr = []
        count = 256*256*3+1
        for i in range(count-1):
            x1 = (self.a * self.x0 + self.c) % self.m
            if(i!=count-2):
                arr.append(x1)
                self.x0=x1
            else:
                arr.append(x1)
        
        img_key = np.asarray(arr)
        img_key = img_key.reshape(256,256,3)
        
        return img_key