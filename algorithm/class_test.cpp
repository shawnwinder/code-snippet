#include <iostream>
using namespace std;

class A{
public:
    void fun(){
        cout << "in A " << endl;
    }
};

class B{
public:
    void fun(){
        cout << "in B " << endl;
    }
};


class C:public A,public B{
public: 
    void fun(){
        cout << "in C " << endl;
    }
};

int main(int argc, char const *argv[]) { 
    A a;
    C c;
    int d = 0;
    a = c;

    a.fun(); 
    c.fun();
    return 0;
}
