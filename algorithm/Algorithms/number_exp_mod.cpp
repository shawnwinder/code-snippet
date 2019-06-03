#include <iostream>
#include <vector>
#define MOD 1000000007

using namespace std;

long long fast_pow_mod(long long base, long long exp) {
    // edge case & illegal cases
    if (exp == 0) {
        return 1;
    }
    if (exp < 0 && base == 0) {
        cout << "cannot divided by ZERO!" << endl;
        return -1;
    }
    if (exp < 0) {
        cout << "fraction has no definition of MOD" << endl;
        return -2;
    }
    if (base == 0 || base == 1) {
        return base;
    }

    // general cases
    bool neg_base = false;
    if (base < 0) {
        neg_base = true;
        base = -base;
    }

    long long EXP = exp;
    long long res = 1;
    long long factor = base;
    while (exp) {
        if (exp & 1) {
            res = (res * factor) % MOD;
        }
        factor = (factor % MOD) * (factor % MOD) % MOD;
        exp >>= 1;
    }

    if (neg_base && (EXP&1)) {
        return -res;
    }else {
        return res;
    }
}

// consider only positive base and exp
long long fast_pow_mod2(long long base, long long exp) {
    if (exp == 1) {
        return base % MOD;
    }

    long long left = exp >> 1;
    long long right = exp - left;
    long long ans = 1;
    ans = (ans * fast_pow_mod2(base, left)) % MOD;
    ans = (ans * fast_pow_mod2(base, right)) % MOD;

    return ans;
}

// consider only positive base and exp
long long fast_pow_mod3(long long base, long long exp) {
    if (exp == 1) {
        return base % MOD;
    }

    long long ans = 1;
    if (exp & 1) {
        ans = base % MOD;
    }
    long long tmp = fast_pow_mod3(base, exp >> 1) % MOD;
    tmp = (tmp * tmp) % MOD;
    ans = ans * tmp % MOD;

    return ans;
}

int main() {
    long long base = 3;
    long long exp = 4;

    while (true) {
        cout << "please input base and exponent: (base exp)" << endl;
        cin >> base >> exp;
        // cout << "fast pow of base is: " << fast_pow(base, exp) << endl;
        cout << "fast pow of base is: " << fast_pow_mod(base, exp) << endl;
        cout << "fast pow of base 3 is: " << fast_pow_mod3(base, exp) << endl;
        cout << "fast pow of base 2 is: " << fast_pow_mod2(base, exp) << endl;
    }

    return 0;
}
