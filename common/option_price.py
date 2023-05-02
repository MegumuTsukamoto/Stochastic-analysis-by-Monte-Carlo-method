import math
import numpy as np
from scipy import stats

# BS_Model（プレーンバニラ）
# 解析解を用いたオプション価格
class BS_Model:

    def __init__(self, s0, K, r, T, sigma):
        self.s0 = s0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

    def Call(self, s0, K, r, T, sigma):
        self.s0 = s0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

        d1 = (np.log(s0/K)+(r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = s0 * stats.norm.cdf(d1,0,1) - K * np.exp(-r*T) * stats.norm.cdf(d2,0,1)
        return call
    
    def Put(self, s0, K, r, T, sigma):
        self.s0 = s0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

        d1 = (np.log(s0/K)+(r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put =  K * np.exp(-r*T) * stats.norm.cdf(-d2,0,1) - s0 * stats.norm.cdf(-d1,0,1)
        return put

# Euler法によるモンテカルロ法
class BS_Monte_Carlo_Euler:
    def __init__(self, s0, mu, sigma, dt, n, N):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.n = n  # n = T /dt
        self.N = N
    
    def Price_Euler(self, s0, mu, sigma, dt, n=50):
        s = s0
        for i in range(n):
            eps = np.random.normal(0,1,1)
            ds = mu * s * dt + sigma * s * eps * np.sqrt(dt)
            s = s + ds
        return s
    
    def MC_Call_Euler(self, s0, K, r, T, mu, sigma, dt, n, N):
        Payoff = 0
        for i in range(N):
            s = self.Price_Euler(s0, mu, sigma, dt, n)
            dPayoff = max(s-K, 0)
            Payoff = Payoff + dPayoff
        mean_Payoff = Payoff / N
        MC_call = mean_Payoff * np.exp(-r * T)
        return MC_call
    
    def MC_Put_Euler(self, s0, K, r, T, mu, sigma, dt, n, N):
        Payoff = 0
        for i in range(N):
            s = self.Price_Euler(s0, mu, sigma, dt, n)
            dPayoff = max(K-s, 0)   # callの逆
            Payoff = Payoff + dPayoff
        mean_Payoff = Payoff / N
        MC_put = mean_Payoff * np.exp(-r * T)
        return MC_put
    
# 正確な離散化よるモンテカルロ法
class BS_Monte_Carlo:
    def __init__(self, s0, mu, sigma, dt, N):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.N = N
    
    def Price(self, s0, mu, sigma, dt):
        eps = np.random.normal(0,1,1)
        s = s0 * np.exp((mu - (sigma**2) / 2 ) * dt + sigma * eps * np.sqrt(dt))
        return s
    
    def MC_Call(self, s0, K, r, T, mu, sigma, dt, N):
        Payoff = 0
        for i in range(N):
            s = self.Price(s0, mu, sigma, dt)
            dPayoff = max(s-K, 0)
            Payoff = Payoff + dPayoff
        mean_Payoff = Payoff / N
        MC_call = mean_Payoff * np.exp(-r * T)
        return MC_call
    
    def MC_Put(self, s0, K, r, T, mu, sigma, dt, N):
        Payoff = 0
        for i in range(N):
            s = self.Price(s0, mu, sigma, dt)
            dPayoff = max(K-s, 0)   # callの逆
            Payoff = Payoff + dPayoff
        mean_Payoff = Payoff / N
        MC_put = mean_Payoff * np.exp(-r * T)
        return MC_put


# エキゾチック
# キャッシュ・オア・ナッシング・コールの解析解を用いたオプション価格
class Cash_Or_Nothing:
    def __init__(self, s0, X, K, r, t, T, sigma):
        self.s0 = s0
        self.X =X
        self.K = K
        self.r = r
        self.t = t
        self.T = T
        self.sigma = sigma

    def Call(self, s0, X, K, r, t, T, sigma):
        self.s0 = s0
        self.X = X
        self.K = K
        self.r = r
        self.t = t
        self.T = T
        self.sigma = sigma

        d1 = (np.log(s0/K)+(r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = X * np.exp(-r*t) * stats.norm.cdf(d2,0,1)
        return call

# キャッシュ・オア・ナッシング・コールのモンテカルロ法
class Cash_Or_Nothing_Monte_Carlo:
    def __init__(self, s0, X, mu, sigma, dt, N):
        self.s0 = s0
        self.X = X
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.N = N
    
    def Price(self, s0, mu, sigma, dt):
        eps = np.random.normal(0,1,1)
        s = s0 * np.exp((mu - (sigma**2) / 2 ) * dt + sigma * eps * np.sqrt(dt))
        return s
    
    def MC_Call(self, s0, X, K, r, t, T, mu, sigma, dt, N):
        Payoff = 0
        for i in range(N):
            s = self.Price(s0, mu, sigma, dt)

            if s > K:
                dPayoff = X
                Payoff = Payoff + dPayoff

        mean_Payoff = Payoff / N
        MC_call = mean_Payoff * np.exp(-r * t)
        return MC_call
    
# 無配当株式を原資産とする変動ルックバック・コールの解析解を用いたオプション価格
class Lookback:
    def __init__(self, s0, smin, K, r, T, sigma):
        self.s0 = s0
        self.smin = smin
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

    def Call(self, s0, smin, K, r, T, sigma):
        self.s0 = s0
        self.smin = smin
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

        a1 = (np.log(s0/smin)+(r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        a2 = a1 - sigma * np.sqrt(T)
        a3 = (np.log(s0/smin)+(-r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        Y1 = (-2 * (r - sigma**2/2) * np.log(s0/smin)) / (sigma**2)
        call = s0 * stats.norm.cdf(a1,0,1) - s0 * (sigma**2/(2*r)) * stats.norm.cdf(-a1,0,1) - \
        smin * np.exp(-r*T) * (stats.norm.cdf(a2,0,1) - sigma**2/(2*r) * np.exp(Y1) * stats.norm.cdf(-a3,0,1))
        return call

# 変動ルックバック・コールのモンテカルロ法
class Lookback_Monte_Carlo:
    def __init__(self, s0, smin, mu, sigma, T, tstep, N):
        self.s0 = s0
        self.smin = smin
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.tstep = tstep
        self.N = N
    
    def Price(self, s0, mu, sigma, dt):
        eps = np.random.normal(0,1,1)
        s = s0 * np.exp((mu - (sigma**2) / 2 ) * dt + sigma * eps * np.sqrt(dt))
        return s
    
    def MC_Call(self, s0, smin, K, r, T, tstep, mu, sigma, dt, N):
        Payoff = 0
        calc_err_buffer = tstep / 100
        for i in range(N):
            st = s0
            stmin = smin
            t = tstep
            while t <= T + calc_err_buffer:
                st = self.Price(st, mu, sigma, tstep)
                t = t + tstep
                if st < stmin:
                    stmin = st
            dPayoff = max(st-stmin, 0)
            Payoff = Payoff + dPayoff

        mean_Payoff = Payoff / N
        MC_call = mean_Payoff * np.exp(-r * T)
        return MC_call
    
# 無配当株式を原資産とするダウン・アンド・アウト・コールの解析解を用いたオプション価格
class Down_and_Out:
    def __init__(self, s0, K, r, t, T, sigma, B):
        self.s0 = s0
        self.K = K
        self.r = r
        self.t = t
        self.T = T
        self.sigma = sigma
        self.B = B

    def Call(self, s0, K, r, t, T, sigma, B):
        self.s0 = s0
        self.K = K
        self.r = r
        self.t = t
        self.T = T
        self.sigma = sigma
        self.B = B

        lmd = (r + sigma**2 / 2) / (sigma**2)
        d3 = np.log(B**2 / (s0*K)) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
        BS = BS_Model(s0, K, r, t, sigma)
        cBS = BS.Call(s0, K, r, t, sigma)
        call = cBS - s0 * (B/s0)**(2*lmd) * stats.norm.cdf(d3,0,1) \
            + K * np.exp(-r*t) * (B/s0)**(2*lmd-2) * stats.norm.cdf(d3-sigma*np.sqrt(T),0,1)
        return call

# ダウン・アンド・アウト・コールのモンテカルロ法
class Down_and_Out_Monte_Carlo:
    def __init__(self, s0, mu, sigma, B, T, tstep, N):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.B = B
        self.T = T
        self.tstep = tstep
        self.N = N
    
    def Price(self, s0, mu, sigma, dt):
        eps = np.random.normal(0,1,1)
        s = s0 * np.exp((mu - (sigma**2) / 2 ) * dt + sigma * eps * np.sqrt(dt))
        return s
    
    def MC_Call(self, s0, K, r, T, tstep, mu, sigma, B, dt, N):
        Payoff = 0
        exting = 0
        calc_err_buffer = tstep / 100
        for i in range(N):
            t = 0
            st = s0
            maturity_flag = 0
            exting_flag = 0

            while True:
                if st <= B:
                    exting_flag = 1
                else:
                    t = t + tstep

                if t <= T + calc_err_buffer:
                    st = self.Price(st, mu, sigma, tstep)
                else:
                    maturity_flag = 1
                
                if exting_flag or maturity_flag:
                    break

            if exting_flag:
                exting = exting + 1
            else:
                dPayoff = max(st-K, 0)
                Payoff = Payoff + dPayoff

        mean_Payoff = Payoff / N
        MC_call = mean_Payoff * np.exp(-r * t)
        exting_rate = exting / N
        return MC_call, exting_rate
    
# アベレージ・ストライク・コールのモンテカルロ法
class Asian_Monte_Carlo:
    def __init__(self, s0, mu, sigma, mstart, mend, T, tstep, N):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.mstart = mstart
        self.mend = mend
        self.T = T
        self.tstep = tstep
        self.N = N
    
    def Price(self, s0, mu, sigma, dt):
        eps = np.random.normal(0,1,1)
        s = s0 * np.exp((mu - (sigma**2) / 2 ) * dt + sigma * eps * np.sqrt(dt))
        return s
    
    def MC_Call(self, s0, K, r, T, tstep, mu, sigma, mstart, mend, dt, N):
        Payoff = 0
        mK = 0
        calc_err_buffer = tstep / 100
        for i in range(N):
            st = s0
            ms = 0
            pnum = 0
            t = 0
            maturity_flag = 0

            while True:
                if t >= mstart and t <= mend:
                    ms = ms + st
                    pnum = pnum + 1
                t = t + tstep

                if t <= T + calc_err_buffer:
                    st = self.Price(st, mu, sigma, tstep)
                else:
                    maturity_flag = 1
                
                if maturity_flag:
                    break

            ms = ms / pnum
            dPayoff = max(st-ms, 0)
            Payoff = Payoff + dPayoff
            mK = mK + ms

        mean_Payoff = Payoff / N
        MC_call = mean_Payoff * np.exp(-r * t)
        mK = mK / N
        return MC_call, mK