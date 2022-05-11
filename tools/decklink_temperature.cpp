/**
 * @file
 *
 * In Linux, link with DeckLinkAPIDispatch.cpp.
 *
 * See also StatusMonitor.cpp BMD SDK example.
 *
 * This file may be safely removed - nothing depends on it.
 *
 * @todo
 * Now most of the devices doesn't implement temperature interface. When it does,
 * consider merging this to UltraGrid. (Also check usual range and units - some
 * warning about overheating can be issued.)
 */

#include <cstdint>
#include <iostream>

#ifdef WIN32
#include "../ext-deps/DeckLink/Windows/DeckLinkAPI_h.h" /*  From DeckLink SDK */
#else
#include "../ext-deps/DeckLink/Linux/DeckLinkAPI.h" /*  From DeckLink SDK */
#endif

using namespace std;

int main() {
        cout << "Prints temperatures for all BMD devices. Takes no argument.\n\n";
        IDeckLinkIterator *deckLinkIterator = nullptr;
#ifdef WIN32
        // Initialize COM on this thread
        HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(result)) {
                return 2;
        }
        result = CoCreateInstance(CLSID_CDeckLinkIterator, nullptr, CLSCTX_ALL,
                        IID_IDeckLinkIterator, (void **) &deckLinkIterator);
        if (FAILED(result)) {
                return 2;
        }
#else
        deckLinkIterator = CreateDeckLinkIteratorInstance();
#endif
        if (deckLinkIterator == nullptr) {
                return 1;
        }

        int numDevices = 0;
        // Enumerate all cards in this system
        IDeckLink *deckLink = nullptr;
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                IDeckLinkStatus *stat = nullptr;
                deckLink->QueryInterface(IID_IDeckLinkStatus, reinterpret_cast<void**>(&stat));
                int64_t temp;
                cout << "Device " << numDevices++ << " temperature: ";
                if (HRESULT res = stat->GetInt(bmdDeckLinkStatusDeviceTemperature, &temp); res != S_OK) {
                        if (res == E_NOTIMPL) {
                                cout << "NOT IMPLEMENTED\n";
                        } else {
                                cout << "error: 0x" << hex << "\n";
                        }
                } else {
                        cout << temp << "\n";
                }

                deckLink->Release();
        }
        if (numDevices == 0) {
                cout << "\tno devices found!\n";
        }

        deckLinkIterator->Release();

#ifdef WIN32
        CoUninitialize();
#endif
}

