
import java.io.FileInputStream;
import java.io.Reader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.*;

/**
 * Tests Scanner.java by running the tokenizer on ScannerTest.txt or 
 * ScannerTestAdvanced.txt.
 *
 * @author Elijah Levanon
 * @version 2025-01-23
 */
public class ScannerTester
{
    /**
     * Initializes a scanner and runs it on ScannerTest.txt or
     * ScannerTestAdvanced.txt, printing each of the tokens extracted 
     * until the end of file is reached.
     * 
     * @param args command line arguments
     */
    public static void main (String[] args) throws IOException
    {
        if (args.length != 1)
        {
            System.out.println("Usage: java ScannerTester <filename>");
            return;
        }

        String fileName = args[0];

        Reader reader;

        try
        {
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(
                    new File(fileName))));
            reader.mark(100000);
            Scanner scanner = new Scanner(reader);
            List<Scanner.Token> tokens = new ArrayList<>(200);
            while (true)
            {
                Scanner.Token nextToken = scanner.nextToken();
                if (nextToken.error.equals("EOF"))
                {
                    break;
                }
                if (!nextToken.error.equals(""))
                {
                    tokens.add(nextToken);
                }
                //System.out.println(scanner.stack);
            }

            int errorNumber = 0;
            int warningNumber = 0;
            for (Scanner.Token t : tokens)
            {
                if (t.sure) errorNumber++; else warningNumber++;
            }

            System.out.printf(
                    "Analysis Complete:\n%d Errors,\n%d Warnings\n",
                    errorNumber, warningNumber);
            System.out.println("==============================");

            tokens.sort((a, b)->(a.line - b.line));

            for (Scanner.Token t : tokens)
            {
                System.out.println(t);
            }
        } 
        catch (FileNotFoundException e)
        {
            System.out.printf("Could not open %s.", fileName);
        }


    }
}
