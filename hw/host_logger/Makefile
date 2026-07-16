# Build wave_logger against the scamp5d interface library (Linux).
#
# Point DEVKIT at the "scamp5d_interface" folder of the devkit package and
# build that library first (open vs_interface.workspace in Codeblocks, build
# the "scamp5d_interface" project - see the devkit README.md). The header
# and library paths below may need adjusting to your devkit layout; on
# Windows use the prebuilt scamp5d_interface.lib with Visual Studio and the
# "scamp5d_interface_app" example project settings instead.
#
# Dependencies (Ubuntu):  sudo apt-get install libuv1-dev libusb-1.0-0-dev
# USB permissions: run the devkit's scamp_usb.sh once, or run via sudo.

DEVKIT ?= $(HOME)/scamp5d/scamp5d_interface

CXX      ?= g++
CXXFLAGS += -std=c++11 -O2 -I$(DEVKIT) -I$(DEVKIT)/include
LDFLAGS  += -L$(DEVKIT) -L$(DEVKIT)/lib -L$(DEVKIT)/bin
LDLIBS   += -lscamp5d_interface -lusb-1.0 -luv -lpthread

wave_logger: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f wave_logger

.PHONY: clean
