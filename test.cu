#include <cstring>
#include <iostream>

#include "test.cuh"


void testCoalesced() {
    std::cout << "Non coalesced access: ";
    test(noCoalescedAccess);
    std::cout << "Coalesced Access: ";
    test(coalescedAccess);
}