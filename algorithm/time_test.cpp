#include <iostream>
#include <ctime>
#include <unistd.h>

using namespace std;

int main() {
    time_t tb;
    time(&tb);

    cout << "tb: " << tb << endl;
    cout << "tb: " << ctime(&tb) << endl;
    sleep(3);

    time_t te;
    time(&te);
    cout << "te: " << te << endl;
    cout << "te: " << ctime(&te) << endl;

    return 0;
}
