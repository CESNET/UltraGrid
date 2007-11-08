
import  wx
import  os, stat

#----------------------------------------------------------------------

PATH2DXT = "/Users/luc/Dev/FastDXT/bin/2dxt"

WIDTH  = 1920
HEIGHT = 1080
FORMAT = 1
#----------------------------------------------------------------------

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def Convert(fn):
   dn, bs = os.path.split(fn)
   file, ext = os.path.splitext(bs)

   if ext != ".rgb" and ext != ".rgba":
      frgb = "%s/%s.%s" % ( dn, file, "rgba" )
      command = "convert %s -depth 8 RGBA:%s" % (fn, frgb)
      print command
      os.system( command )

      fout = "%s/%s.%s" % ( dn, file, "dxt" )
      command = "%s %d %d %d %s %s && /bin/rm -f %s" % (PATH2DXT, WIDTH, HEIGHT, FORMAT, frgb, fout, frgb)
   else:
      fout = "%s/%s.%s" % ( dn, file, "dxt" )
      command = "%s %d %d %d %s %s" % (PATH2DXT, WIDTH, HEIGHT, FORMAT, fn, fout)

   print command
   os.system( command )

class MyFileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):
        self.window.SetInsertionPointEnd()
        self.window.Clear()

        fn = list()

        if len(filenames) == 1:
           mode = os.stat(filenames[0])[stat.ST_MODE]
           if stat.S_ISDIR(mode):
              # It's a directory, recurse into it
              for f in mylistdir(filenames[0]):
                 full = os.path.join(filenames[0], f)
                 if stat.S_ISREG( os.stat(full)[stat.ST_MODE] ):
                    fn.append(full)
           elif stat.S_ISREG(mode):
              # It's a file, call the callback function
              fn.append(filenames[0])
           else:
              # Unknown file type, print a message
              print 'Skipping %s' % pathname
        else:
           for f in filenames:
              fn.append(f)

        fn.sort()
        max = len(fn)

        self.window.WriteText("\n%d file(s) dropped:\n" % len(fn))

        dlg = wx.ProgressDialog("DXT Conversion", "Converting RAW files to DXT",
                                maximum = max,
                                parent=self.window,
                                style = wx.PD_CAN_ABORT | wx.PD_APP_MODAL| wx.PD_ELAPSED_TIME| wx.PD_REMAINING_TIME)
            
        keepGoing = True
        count = 0
            
        while keepGoing and count < max:
           self.window.WriteText(fn[count] + '\n')
           (keepGoing, skip) = dlg.Update(count, fn[count])
           Convert(fn[count])
           count += 1

        dlg.Destroy()
       

class FileDropPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Format selection
        rb = wx.RadioBox(self, -1, "Select DXT format:", wx.DefaultPosition, wx.DefaultSize,
                         ['DXT1', 'DXT5', 'DXT6 (5-YCoCg)'], 1, wx.RA_SPECIFY_ROWS)
        self.Bind(wx.EVT_RADIOBOX, self.OnFormat, rb)
        sizer.Add(rb, 0, wx.ALL, 5)

        # Width and height setting
        ssizer = wx.BoxSizer(wx.HORIZONTAL)

        ssizer.Add(wx.StaticText(self, -1, "Width", style=wx.ALIGN_BOTTOM), 0, wx.ALL, 5)
        sw = wx.SpinCtrl(self, -1, "",  wx.DefaultPosition, (100,-1))
        sw.SetRange(1,65535)
        sw.SetValue(WIDTH)
        self.Bind(wx.EVT_SPINCTRL, self.OnWidth, sw)
        ssizer.Add(sw, 0, wx.ALL, 0)

        ssizer.Add(wx.StaticText(self, -1, "Height", style=wx.ALIGN_BOTTOM), 0, wx.ALL, 5)
        sh = wx.SpinCtrl(self, -1, "",  wx.DefaultPosition, (100,-1))
        sh.SetRange(1,65535)
        sh.SetValue(HEIGHT)
        self.Bind(wx.EVT_SPINCTRL, self.OnHeight, sh)
        ssizer.Add(sh, 0, wx.ALL, 0)

        sizer.Add(ssizer, 0, wx.ALL, 5)

        # Drop box
        sizer.Add(
            wx.StaticText(self, -1, " \nDrag some files here:"),
            0, wx.EXPAND|wx.ALL, 5)

        self.text = wx.TextCtrl(
                        self, -1, "\n\tImage files ...",
                        style = wx.TE_MULTILINE|wx.HSCROLL|wx.TE_READONLY)

        dt = MyFileDropTarget(self)
        self.text.SetDropTarget(dt)
        sizer.Add(self.text, 1, wx.EXPAND)
        
        self.SetAutoLayout(True)
        self.SetSizer(sizer)

    def OnWidth(self, arg):
       global WIDTH
       WIDTH = arg.GetInt()
    def OnHeight(self, arg):
       global HEIGHT
       HEIGHT = arg.GetInt()
    def OnFormat(self, arg):
       global FORMAT
       f = arg.GetInt()
       if f == 0:
          FORMAT = 1
       if f == 1:
          FORMAT = 5
       if f == 2:
          FORMAT = 6

    def WriteText(self, text):
        self.text.WriteText(text)

    def Clear(self):
        self.text.Clear()

    def SetInsertionPointEnd(self):
        self.text.SetInsertionPointEnd()


#----------------------------------------------------------------------

class TestPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self.SetAutoLayout(True)
        outsideSizer = wx.BoxSizer(wx.VERTICAL)

        msg = "DXT Conversion Drag-And-Drop"
        text = wx.StaticText(self, -1, "", style=wx.ALIGN_CENTRE)
        text.SetFont(wx.Font(24, wx.SWISS, wx.NORMAL, wx.BOLD, False))
        text.SetLabel(msg)

        w,h = text.GetTextExtent(msg)
        text.SetSize(wx.Size(w,h+1))
        text.SetForegroundColour(wx.BLUE)
        outsideSizer.Add(text, 0, wx.EXPAND|wx.ALL, 5)
        outsideSizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND)

        inSizer = wx.BoxSizer(wx.HORIZONTAL)
        inSizer.Add(FileDropPanel(self), 1, wx.EXPAND)

        outsideSizer.Add(inSizer, 1, wx.EXPAND)
        self.SetSizer(outsideSizer)


#----------------------------------------------------------------------
class MainWindow(wx.Frame):
   """ This window displays the GUI Widgets. """
   def __init__(self,parent,id,title):
       wx.Frame.__init__(self,parent,-4, title, size = (500,300), style=wx.DEFAULT_FRAME_STYLE|wx.NO_FULL_REPAINT_ON_RESIZE)
       self.SetBackgroundColour(wx.WHITE)

       w = TestPanel(self)
       
       # Display the Window
       self.Show(True)


class MyApp(wx.App):
   """ DXT Conversion """
   def OnInit(self):
      """ Initialize the Application """
      # Declare the Main Application Window
      frame = MainWindow(None, -1, "DXT Drag and Drop")

      # Show the Application as the top window
      self.SetTopWindow(frame)
      return True


def runTest(frame, nb, log):
    win = TestPanel(nb, log)
    return win

# Declare the Application and start the Main Loop
app = MyApp(0)
app.MainLoop()
