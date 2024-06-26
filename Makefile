#############################################################################
# Makefile for building: cs4670
# Generated by qmake (2.01a) (Qt 4.6.2) on: Wed Sep 8 17:10:01 2010
# Project:  cs4670.pro
# Template: app
# Command: /opt/NokiaQtSDK/Maemo/4.6.2/targets/fremantle-1030/bin/qmake -unix -o Makefile cs4670.pro
#############################################################################

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DQT_GL_NO_SCISSOR_TEST -DQT_DEFAULT_TEXTURE_GLYPH_CACHE_WIDTH=1024 -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED
CFLAGS        = -pipe -g -D_REENTRANT -Wall -W $(DEFINES)
CXXFLAGS      = -pipe -g -D_REENTRANT -Wall -W $(DEFINES)
INCPATH       = -I/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/linux-g++-maemo5 -I. -I/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/include/QtCore -I/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/include/QtGui -I/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/include -I.
LINK          = g++
LFLAGS        = -Wl,-rpath-link,/usr/lib -Wl,-rpath,/usr/lib
LIBS          = $(SUBLIBS)  -L/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/lib -Llib -ljpeg -lFCam -lcv -lcvaux -lhighgui -lcxcore -lQtGui -L/usr/lib -L/usr/X11R6/lib -lQtCore -lpthread 
AR            = ar cqs
RANLIB        = 
QMAKE         = /opt/NokiaQtSDK/Maemo/4.6.2/targets/fremantle-1030/bin/qmake
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = $(COPY)
COPY_DIR      = $(COPY) -r
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = N900Main.cpp \
		OverlayWidget.cpp \
		CameraThread.cpp \
		Framebuffer.cpp \
		N900Helpers.cpp \
		CameraWidget.cpp \
		Filters.cpp moc_CameraThread.cpp \
		moc_CameraWidget.cpp
OBJECTS       = N900Main.o \
		OverlayWidget.o \
		CameraThread.o \
		Framebuffer.o \
		N900Helpers.o \
		CameraWidget.o \
		Filters.o \
		moc_CameraThread.o \
		moc_CameraWidget.o
DIST          = /opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/common/unix.conf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/common/linux.conf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/qconfig.pri \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt_config.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/default_pre.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/debug.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/default_post.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/unix/thread.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/moc.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/warn_on.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/resources.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/uic.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/yacc.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/lex.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		cs4670.pro
QMAKE_TARGET  = cs4670
DESTDIR       = 
TARGET        = cs4670

first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

Makefile: cs4670.pro  /opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/linux-g++-maemo5/qmake.conf /opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/common/unix.conf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/common/linux.conf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/qconfig.pri \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt_config.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/default_pre.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/debug.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/default_post.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/unix/thread.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/moc.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/warn_on.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/resources.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/uic.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/yacc.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/lex.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/lib/libQtGui.prl \
		/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/lib/libQtCore.prl
	$(QMAKE) -unix -o Makefile cs4670.pro
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/common/unix.conf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/common/linux.conf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/qconfig.pri:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt_functions.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt_config.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/exclusive_builds.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/default_pre.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/debug.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/default_post.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/qt.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/unix/thread.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/moc.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/warn_on.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/resources.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/uic.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/yacc.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/lex.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/share/qt4/mkspecs/features/include_source_dir.prf:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/lib/libQtGui.prl:
/opt/NokiaQtSDK/Maemo/4.6.2/sysroots/fremantle-arm-sysroot-1030-slim/usr/lib/libQtCore.prl:
qmake:  FORCE
	@$(QMAKE) -unix -o Makefile cs4670.pro

dist: 
	@$(CHK_DIR_EXISTS) .tmp/cs46701.0.0 || $(MKDIR) .tmp/cs46701.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) .tmp/cs46701.0.0/ && $(COPY_FILE) --parents OverlayWidget.h CameraThread.h Framebuffer.h N900Helpers.h CameraWidget.h Filters.h .tmp/cs46701.0.0/ && $(COPY_FILE) --parents N900Main.cpp OverlayWidget.cpp CameraThread.cpp Framebuffer.cpp N900Helpers.cpp CameraWidget.cpp Filters.cpp .tmp/cs46701.0.0/ && (cd `dirname .tmp/cs46701.0.0` && $(TAR) cs46701.0.0.tar cs46701.0.0 && $(COMPRESS) cs46701.0.0.tar) && $(MOVE) `dirname .tmp/cs46701.0.0`/cs46701.0.0.tar.gz . && $(DEL_FILE) -r .tmp/cs46701.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

compiler_moc_header_make_all: moc_CameraThread.cpp moc_CameraWidget.cpp
compiler_moc_header_clean:
	-$(DEL_FILE) moc_CameraThread.cpp moc_CameraWidget.cpp
moc_CameraThread.cpp: Filters.h \
		CameraThread.h
	/opt/NokiaQtSDK/Maemo/4.6.2/targets/fremantle-1030/bin/moc $(DEFINES) $(INCPATH) CameraThread.h -o moc_CameraThread.cpp

moc_CameraWidget.cpp: CameraThread.h \
		Filters.h \
		CameraWidget.h
	/opt/NokiaQtSDK/Maemo/4.6.2/targets/fremantle-1030/bin/moc $(DEFINES) $(INCPATH) CameraWidget.h -o moc_CameraWidget.cpp

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_image_collection_make_all: qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) qmake_image_collection.cpp
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_uic_make_all:
compiler_uic_clean:
compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: compiler_moc_header_clean 

####### Compile

N900Main.o: N900Main.cpp OverlayWidget.h \
		CameraWidget.h \
		CameraThread.h \
		Filters.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o N900Main.o N900Main.cpp

OverlayWidget.o: OverlayWidget.cpp OverlayWidget.h \
		Framebuffer.h \
		omapfb.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o OverlayWidget.o OverlayWidget.cpp

CameraThread.o: CameraThread.cpp CameraThread.h \
		Filters.h \
		OverlayWidget.h \
		Framebuffer.h \
		omapfb.h \
		N900Helpers.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o CameraThread.o CameraThread.cpp

Framebuffer.o: Framebuffer.cpp Framebuffer.h \
		omapfb.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Framebuffer.o Framebuffer.cpp

N900Helpers.o: N900Helpers.cpp N900Helpers.h \
		Framebuffer.h \
		omapfb.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o N900Helpers.o N900Helpers.cpp

CameraWidget.o: CameraWidget.cpp CameraWidget.h \
		CameraThread.h \
		Filters.h \
		OverlayWidget.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o CameraWidget.o CameraWidget.cpp

Filters.o: Filters.cpp Filters.h \
		Project1.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Filters.o Filters.cpp

moc_CameraThread.o: moc_CameraThread.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_CameraThread.o moc_CameraThread.cpp

moc_CameraWidget.o: moc_CameraWidget.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_CameraWidget.o moc_CameraWidget.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

