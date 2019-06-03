void foo(){;}

int main(){
    int i = 10;
    foo();

    int b;
    foo();

    b = 10;
    foo();

    int *p = &i;
    foo();

    int &s = i;
    foo();

    (*p) = 8;
    foo();

    s = 0;
    foo();

    int *const p_self = &i;
    const int* p_int = &i;

    return 0;
}
