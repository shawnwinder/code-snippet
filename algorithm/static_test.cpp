#include <iostream>
using namespace std;

void foo() {
    static int a = 1;
    cout << "a in foo: " << a << endl;
    cout << "b in foo: " << b << endl;
}

void bar() {
    static int b = 2;
    cout << "b in bar: " << b << endl;
}

int main () {
    static int c = 3;
    cout << "c in main: " << c << endl;

    foo();
    bar();

    return 0;
}
