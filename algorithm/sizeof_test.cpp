#include <iostream>

using namespace std;

struct A {
    char c; // 1, 00,00,00,00
    int i;  // 4
    short s; // 2,
};

struct B {
    int i; // 4
    char c; // 10,00,00,00
    short s; // 10,02,00,00
};

int main() {
    cout << "sizeof(A): " << sizeof(A) << endl;
    cout << "sizeof(B): " << sizeof(B) << endl;

    return 0;
}
