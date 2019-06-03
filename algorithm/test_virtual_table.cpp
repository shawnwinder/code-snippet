#include <iostream>
using namespace std;


class Base {
    int m_tag;
public:
    Base(int tag) : m_tag(tag) {}
    void print() {
        cout << "Base::print() called" << endl;
    }

    virtual void vPrint() {
        cout << "Base::vPrint() called" << endl;
    }
                        
    virtual void printTag() {
        cout << "Base::m_tag of this instance is: " << m_tag << endl;
    }
};

class Derived : public Base {
public:
    Derived(int tag) : Base(tag) {}
                
    void print() {
        cout << "Derived1::print() called" << endl;
    }
                    
    virtual void vPrint() {
        cout << "Derived::vPrint() called" << endl;
    }
};

int main() {
    Derived* foo = new Derived(1);
    Base* bar = foo;

    foo->print();
    foo->vPrint();

    bar->print();
    bar->vPrint();

    return 0;
}


