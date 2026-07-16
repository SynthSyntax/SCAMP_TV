# Build wave_logger by compiling the scamp5d interface library sources
# directly (no prebuilt .a, no Code::Blocks). Point SCAMP at the devkit's
# "scamp5d_interface" folder - the one holding scamp5d_usb.h and the .cpp
# sources.
#
# Dependencies (Ubuntu):  sudo apt-get install libuv1-dev libusb-1.0-0-dev
# USB permissions: run the devkit's scamp_usb.sh once, or run via sudo.
#
#   make                     # uses the SCAMP path below
#   make SCAMP=/path/to/scamp5d_interface_lib_linux/scamp5d_interface

SCAMP ?= /home/bharghav/scamp5/dev_kit/scamp5d_interface_lib_linux/scamp5d_interface

CXX ?= g++
CC  ?= gcc
CPPFLAGS += -I$(SCAMP)
CXXFLAGS += -std=c++14 -O2
CFLAGS   += -O2
LDLIBS   += -lusb-1.0 -luv -lpthread

# All library sources EXCEPT the Windows USB backend (we use libusb on Linux)
# and the devkit's own demo main (it defines main(), which collides with ours).
LIB_CPP := $(filter-out $(SCAMP)/scamp5d_usb.usbk.cpp $(SCAMP)/debug_main.cpp, \
                        $(wildcard $(SCAMP)/*.cpp))
LIB_C   := $(wildcard $(SCAMP)/debug_functions.c)

BUILD := build
OBJ := $(BUILD)/main.o \
       $(patsubst $(SCAMP)/%.cpp,$(BUILD)/%.o,$(LIB_CPP)) \
       $(patsubst $(SCAMP)/%.c,$(BUILD)/%.o,$(LIB_C))

wave_logger: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/main.o: main.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/%.o: $(SCAMP)/%.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/%.o: $(SCAMP)/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) wave_logger

.PHONY: clean
