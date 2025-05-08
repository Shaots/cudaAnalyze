#include <cstring>
#include <iostream>

#include "test.cuh"

void testCoalesced() {
    std::cout << "Non coalesced access: ";
    test(noCoalescedAccess);
    std::cout << "Coalesced Access: ";
    test(coalescedAccess);
}

void testMatrix() {
    std::cout << "Simple multiply ";
    testMultiply(simpleMultiply);

    std::cout << "Coalesced multiply ";
    testMultiply(coalescedMultiply);

    std::cout << "Shared multiply ";
    testMultiply(sharedMultiply);
}