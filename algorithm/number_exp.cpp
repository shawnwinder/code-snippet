#include <iostream>
#include <vector>

using namespace std;

// NOTE: code below consider no overflow or underflow
double fast_pow(int base, int exp) {
    // edge cases
    if (base == 0 && exp > 0) {
        return 0;
    }
    if (base == 0 && exp < 0) {
        cout << "wrong input" << endl;
        return -1;
    }
    if (base == 1 || exp == 0) { 
        return 1;
    }

    // general cases
    bool neg = false;
    if (exp < 0) {
        neg = true;
        exp = -exp;
    }

    int ans = 1;
    int factor = base;
    while (exp) {
        if (exp & 1) {
            ans *= factor;
        }
        factor *= factor;
        exp >>= 1;
    }

    return neg ? 1.0 / ans : 1.0 * ans;
}

int main() {
    int base = 3;
    int exp = 4;

    while (true) {
        cout << "please input base and exponent: (base exp)" << endl;
        cin >> base >> exp;
        cout << "fast pow of base is: " << fast_pow(base, exp) << endl;
    }

    return 0;
}
