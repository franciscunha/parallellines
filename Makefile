SYSCONF_LINK = g++
CPPFLAGS     =
LDFLAGS      =
LIBS         = -lm

BUILDDIR = build
DESTDIR  = $(BUILDDIR)/out
OBJDIR 	 = $(BUILDDIR)/objects
SRCDIR 	 = src
TARGET   = main

OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(wildcard $(SRCDIR)/*.cpp))

all: $(OBJDIR) $(DESTDIR)/$(TARGET)

$(OBJDIR):
	mkdir "$(BUILDDIR)\out"
	mkdir "$(BUILDDIR)\objects"

$(DESTDIR)/$(TARGET): $(OBJECTS)
	$(SYSCONF_LINK) -Wall $(LDFLAGS) -o $@ $(OBJECTS) $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(SYSCONF_LINK) -Wall $(CPPFLAGS) -c $< -o $@

clean:
	del /s /q $(BUILDDIR)
