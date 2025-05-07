#include <iostream>

#include "device_launch_parameters.h"
#include "info.cuh"
#include "test.cuh"

int main(int argc, char* argv[]) {
    printInfoDevice();
    testCoalesced();
    return 0;
}
