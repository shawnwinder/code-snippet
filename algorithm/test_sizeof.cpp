#include <iostream>
using namespace std;


class A {
    static void foo() {}
    friend void nima();
public:
    void bar(){}
    A(){}
    ~A(){}
    virtual void v1(){}
    virtual void v2(){}
    virtual void v3(){}
    virtual void v4(){}
    virtual void v5(){}
    virtual void v6(){}
    virtual void v7(){}
    virtual void v8(){}
    virtual void v9(){}
    virtual void v10(){}
};

void nima(){}

int main () {
    cout << sizeof(A) << endl;

    return 0;
}
