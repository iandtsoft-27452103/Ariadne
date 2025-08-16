using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class Communicate
    {
        public Process prc = new Process();
        public StreamWriter sw;
        public StreamReader sr;

        public Communicate()
        {
            string AppPath = System.AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            string Engine = "start.bat";
            string FilePath = AppPath + "python_src\\" + Engine;

            Process prc = new Process();

            prc.StartInfo.FileName = FilePath;
            prc.StartInfo.Arguments = String.Empty;
            prc.StartInfo.WorkingDirectory = Path.GetDirectoryName(FilePath);
            prc.StartInfo.UseShellExecute = false;
            prc.StartInfo.CreateNoWindow = true;
            prc.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            prc.StartInfo.RedirectStandardInput = true;
            prc.StartInfo.RedirectStandardOutput = true;
            prc.StartInfo.RedirectStandardError = false;

            prc.Start();

            sw = prc.StandardInput;
            sr = prc.StandardOutput;
        }

        public void ThrowRequest(string msg)
        {
            sw.WriteLine(msg);
            sw.Flush();
        }

        public void Boot()
        {
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                if (line == "Hello World.") { break; }
            }
        }

        public string ReceiveResponse()
        {
            try
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    if (line != "") { break; }
                }

                return line;
            }
            catch (Exception e)
            {
                throw;
            }
        }

        public void Quit()
        {
            sw.WriteLine("quit");
            sw.Flush();
        }

        public void Dispose()
        {
            sw.Dispose();
            sr.Dispose();
            prc.Dispose();
        }
    }
}
