#include <iostream>
#include <cmath>

using namespace std;

void compute_pi_1() {
    float pi = 2;
    int i;
    for (i = 100; i >= 1; -- i) {
        pi = pi * (float)i / (2*i + 1) + 2;
    }
    cout << "pi: " << pi << endl;
}

void compute_pi_2() {
    const int LIMIT = 10000000;
    int sign = 1;
    double pi = 0;
    for (int i = 1; i <= LIMIT; ++ i) {
        pi += sign * 1.0 / (2 * i - 1);
        sign *= -1;
    }
    cout << "pi: " << pi * 4 << endl;
}

int main() {
    // compute_pi_1();
    compute_pi_2();

    return 0;
}
