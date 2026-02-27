import java.util.Arrays;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.Stack;
import java.lang.Integer;
import java.io.IOException;

/**
 * This scanner checks code against Dr. Nelson's style guide for C and
 * java, as far as is possible for static analysis.
 * @author Elijah Levanon
 * @author Samuel Tong
 * @version 2026-02-08
 * RULES:
 * 1.  No uncommented block longer than 10 lines.
 * 2.  Comment following a block should be properly indented.
 * 3.  All indentation should be correct (3 spaces, no tabs).
 * 4.  Block comments must be delimited with asterisks.
 * 5.  Block comments must not be indented or consistently indented 
 *     once.
 * 6.  Lines may not be longer than 132 lines.
 * 7.  No magic numbers (basic test - a number not 0, 1, 0.0, 1.0 not
 *     preceded anywhere by " final " or "#define ").
 * 8.  Variable names should be lower camel case or upper snake case
 *     (required if the variable is detected to be a constant).
 * 9.  Variables should not be called "l" or "O".
 * 10. Variables should not be called "i", "j", or "k" outside of a for 
 *     loop and should not be made up of one capital letter.
 * 11. Class names should be upper camel case.
 * 12. There should be no whitespace lines in the interior of a brace
 *     blocks and no whitespace preceding an open brace.
 * 13. There should be a line of whitespace before the line defining a 
 *     loop or conditional and after the closing brace, if the 
 *     indentation is the same, excepting else statements following if 
 *     statements.
 * 14. For loops should have a space after "for", should have spaces 
 *     after the semicolons (if succeeded by a statement), and should 
 *     have spaces around "=", "<", etc. If, while, and switch 
 *     statements should similarly have spaces between the keyword and
 *     the open parenthes. Functions/methods should not have spaces
 *     beteween the name and the open parenthesis.
 * 15. if (...) ...;
 *     else ... is invalid.
 * 16. Break statements are only valid if they break out of a switch
 *     statement, not a loop.
 * 17. A non-void function must have a return statement as the last
 *     executable line and no other return statements. A void function 
 *     must also have a return statement as the last line if the 
 *     language is detected to be C (in the absence of a class or 
 *     interface definition).
 * 18. All functions (C), methods (java), and classes (java) must have a
 *     block comment preceding them.
 * 
 * 
 * The scanner does not do the following:
 * 1. Mixed Mode Arithmetic Detection
 * 2. Actual Magic Number Detection
 * 3. Detection that a loop iterator is not modified within the loop
 * 4. Semantics of a for loop ("for loops should not simulate while 
 *    loops")
 */

%%

%class Scanner
%unicode
%line
%column
%public
%function nextToken
%type Token
%eofval{
return new Token("EOF", -1, false);
%eofval}

%{
    int indentCount = 0;
    boolean lastLineComplete = true; // true if last line ended with ';'
    int colonChain = -1; // if last line ended with a colon (as
                         // in a switch statement's case block)
                         // 3 + the indentation of the line of
                         // the colon becomes a valid indentation
                         // for all subsequent lines (until the
                         // chain is broken), becoming
                         // conditionally permissible along with
                         // the true indentation (3 + the
                         // indentation of the switch statement's
                         // header).

    public void resetFileReading() throws IOException
    {
        zzReader.reset();  // resets the input stream to the correct 
                           // location
        yyreset(zzReader); // sets all internal counts to zero
        zzRefill();        // fills buffer with initial values 
                           // (necessary because jflex doesn't 
                           // ostensibly support resetting the reader
                           // to the start
    }

    Stack<Integer> stack = new Stack<>();

    boolean isJava = false;

    public static class Token
    {
        public final String error;
        public final int line;
        public final boolean sure;

        public static final Token NULL = new Token("", -1, false);

        public Token (String error, int line, boolean sure)
        {
            this.error = error;
            this.line = line;
            this.sure = sure;
        }

        public String toString()
        {
            String s = "[line " + line + "] ";
            if (sure)
            {
                s += "\033[31m";
            }
            else
            {
                s += "\033[33m";
            }
            s += error;
            s += "\033[39m";
            return s;
        }
    }
%}


LT = (\n|\r\n) // line terminator
WS = ([ \t\f]) // white space
//TK = ([^ \t\f\r\n]) // token character
NU = ([0-9]) // numeral
OP = ([=\-/+\[\]\(\)*&\^%!~?:;]) // operand
ID = (([A-Za-z_][A-Za-z0-9_]*)|(\#define)) // identifier
EM = (({WS}*(\/\*.*\*\/)?)*(\/\/.*)?{LT}) // empty line
FN = ((({ID}{WS}+)?{ID}{WS}+)?{ID}{WS}+{ID}{WS}*\(([^{)]*{LT})*[^{)]*\){LT}{WS}*\{) // function
CL = (public|private){WS}+(abstract{WS}+)?(static{WS}+)?(final{WS}+)?(class|interface){WS}+[A-Za-z0-9]+ // class



/*

*/

// handles indenting and multiline comments
%state PASS1 
// handles brackets and singleline comments
%state PASS2
// handles max line count
%state PASS3
// handles magic numbers
%state PASS4
// handles variable names and white space (for, while, if, switch)
%state PASS5
// handles "break" and return detection
%state PASS6

%%

// all blocks containing more than 10 lines

<YYINITIAL> {LT}|.   {
    System.out.println("Starting Pass 1");
    yypushback(1);
    yybegin(PASS1);
}

// ideal multiline comment form
<PASS1> ^.*\/\*.*({LT}{WS}*[ ]?\*.*)*{LT}{WS}*[ ]?\*\/ {
    String text = yytext();
    String[] lines = text.split("\n");
    boolean isNotIndented = true;
    boolean isIndentedOnce = true;
    for (int i = 0; i < lines.length; i++)
    {
        switch(lines[i].indexOf('*'))
        {
            case 0:
            case 1:
                    isIndentedOnce = false; 
                    break;
            case 3:
            case 4:
                    isNotIndented = false; 
                    break;
            default: isIndentedOnce = false; isNotIndented = false; 
                    break;
        }
    }


    if (!isNotIndented && !isIndentedOnce)
    {
        return new Token("Block comment is inconsistently or improperly"
            + "indented", yyline + 1, true);
    }

    //System.out.printf("%s\n", yytext());
    }

// default multiline comment form
<PASS1> ^.*(\/\*)~(\*\/) {
    return new Token("Block comment does not have asterisks on each"
        + " line", yyline + 1, true);
}

// processes logic for indenting
<PASS1> ^.+$ { 
    String text = yytext();
    //System.out.printf("1 TEXT reached: %s\n", Arrays.toString(text.getBytes()));
    String[] parts = text.split("[^ \t\f]", 2);
    if (parts.length < 2) return Token.NULL;
    String indent = parts[0];
    String trueString = text.substring(indent.length());
    parts[1] = trueString;
    //System.out.printf("%s - %s\n", Arrays.toString(parts), 
    //    lastLineComplete ? "true" : "false");

    int savedIndentCount = indentCount;
    boolean savedLastLineComplete = lastLineComplete;

    trueString = trueString.split("//")[0];
    
    if (indent.length() != colonChain)
    {
        colonChain = -1;
    }

    if (trueString.length() > 0 && trueString.charAt(0) == '}')
    {
        indentCount--;
        savedIndentCount--;
        lastLineComplete = true;
    }
    else if (trueString.matches("\\{[ \t\f]*"))
    {
        indentCount++;
        lastLineComplete = true;
    }
    else if (trueString.matches(".*;[ \t\f]*"))
    {
        lastLineComplete = true;
    }
    else if (trueString.matches(".*:[ \t\f]*"))
    {
        savedIndentCount--;
        savedLastLineComplete = false;
        lastLineComplete = true;
        colonChain = indent.length() + 3;
    }
    else
    {
        lastLineComplete = false;
    }
    
    if (indent.indexOf("\t") != -1)
    {
        return new Token("Indent contains tab(s)", yyline + 1, true);
    }
    else
    {
        if (!savedLastLineComplete)
        {
            if (indent.length() < 3 * savedIndentCount &&
                indent.length() != colonChain)
            {
                return new Token("Incorrect indentation: is " 
                    + indent.length() + ", should be at least " + 3 
                    * savedIndentCount + " spaces", yyline + 1, true);
            }
        }
        else if (indent.length() != 3 * savedIndentCount &&
                indent.length() != colonChain)
        {
            return new Token("Incorrect indentation: is " 
                + indent.length() + ", should be " + 3 
                * savedIndentCount + " spaces", yyline + 1, true);
        }
    }
    }

<PASS1> {LT}|. {}

<PASS1> <<EOF>> {
    System.out.println("Starting Pass 2");
    resetFileReading();
    yybegin(PASS2);
    return Token.NULL;
    }

// braces of longer than 10 lines
<PASS2> {LT}.*{LT}.*\{([^}]*{LT}){12}([^}]*{LT})*\}.*$     {
    String text = yytext();
    text = text.substring(text.indexOf('\n') + 1);
    //System.out.println("STRING:\n" + text + "\nEND");
    String[] lines = text.split("\n|\r\n");
    //System.out.println(Arrays.toString(lines));
    String firstLine = lines[0];
    if (!lines[1].matches("^[ \t\f]*\\{[ \t\f]*$"))
    { 
        // returns everything but the opening brace
        yypushback(yylength() - 1);
        return new Token("Opening brace must be on its own line", 
            yyline + 3, true);
    }
    String lastLine = lines[lines.length - 1];
    //System.out.printf("lastLine: %s\n", lastLine);
    //System.out.printf("%s ~ \n%s\n", 
    //Arrays.toString(lastLine.getBytes()),
    //Arrays.toString(("} // " + Pattern.quote(firstLine)).
    //  getBytes()));
    if (!firstLine.matches("^[ \t\f]*$") &&
        !lastLine.matches("} // " + Pattern.quote(firstLine) + 
            "[ \t\f]*$"))
    {
        yypushback(yylength() - 1);
        return new Token("A Block of more than 10 lines has no comment"
            + " or it is improperly placed or formatted", yyline + 2,
            true);
    }
    //System.out.println(Arrays.toString(lines));
    }

// single line comments
<PASS2> ^.*\/\/.*$    {
    String text = yytext();
    if (!text.matches(".*[^ \t\f](.*)//.*"))
    {
        return new Token("Single line comment on its own line; should be converted to a block comment", yyline + 2, true);
    }
    }


<PASS2> {LT}|.    {}

<PASS2> <<EOF>> {
    System.out.println("Starting Pass 3");
    resetFileReading();
    yybegin(PASS3);
    return Token.NULL;
    }

<PASS3> ^.+$    {
    String text = yytext();
    //System.out.println("length: " + text.length());
    if (text.length() > 132)
    {
        return new Token("Line exceeds 132 lines", yyline + 2, true);
    }
}

<PASS3> {LT}|.  {}

<PASS3> <<EOF>> {
    System.out.println("Starting Pass 4");
    resetFileReading();
    yybegin(PASS4);
    return Token.NULL;
}

// ignore numbers in comments
<PASS4> (\/\*)~(\*\/) {}

<PASS4> ^.*\/\/.*$ {}

// ignore numbers in strings
<PASS4> \"([^\"]|(\\\"))*\" {}

// magic numbers
<PASS4> ^.*({OP}|{WS}){NU}+(\.{NU}+)?({OP}|{WS}).*$ {
    String text = yytext();
    //System.out.println(text);
    Matcher matcher = Pattern.compile(
        "[=\\-/+\\[\\]\\(\\)*&^%!~?:; \\t\\f][0-9]+(\\.[0-9]+)?[=\\-/+"
        + "\\[\\]\\(\\)*&^%!~?:; \\t\\f]")
        .matcher(text);
    while (matcher.find())
    {
        int startIndex = matcher.start() + 1;
        int endIndex = matcher.end() - 1;
        String before = text.substring(0, startIndex);
        String match = text.substring(startIndex, endIndex);
        //System.out.printf("[[\"%s\"-\"%s\"]]\n", before, match);
        if (!match.matches("0|1|0.0|1.0") && 
            !before.matches("^((#define |final )|(.*( final ))).*$"))
        {
            return new Token("Potential magic number", yyline + 1,
                false);
        }
    }
    }

<PASS4> {LT}|.  {}

<PASS4> <<EOF>> {
    System.out.println("Starting Pass 5");
    resetFileReading();
    yybegin(PASS5);
    return Token.NULL;
    }

// ignore variables in comments
<PASS5> (\/\*)~(\*\/) {
    String length = yytext();
    String[] lines = length.split("\\r\\n|\\n");
    yypushback(Math.min(lines[lines.length-1].length() + 2, yylength() - 1));
    }

<PASS5> \/\/.*$ {
}

// ignore variables in strings
<PASS5> \"([^\"]|(\\\"))*\" {
}

<PASS5> ^{WS}*if{WS}*\([^)]*\).*;{WS}*(\/\/.*)?{LT}{WS}*else  {
    return new Token("If/else statement is formatted in an improper"
        + " way", yyline + 1, true);
    }

// for loop
<PASS5>^{WS}*for{WS}*\(.*;.*;.*\){WS}*$ {
    String token = "[A-Za-z0-9_]+";
    String operator = "(>|<|>=|<=|==)";
    String text = yytext();
    text = text.substring(text.indexOf('f'));
    //System.out.printf("TEXT: %s\n", text);
    if (text.matches( // attempt to match to most common format
        String.format(
            "for[ ]*\\(%s[ ]*%s[ ]*=[ ]*%s[ ]*;[ ]*%s[ ]*%s[ ]*%s[ ]*;"
            + "[ ]*(\\+\\+[ ]*%s|%s[ ]*\\+\\+|--[ ]*%s|%s[ ]*--)[ ]*"
            + "\\)",
            token, token, token, token, operator, token, token, token,
            token, token)))
    {
        if (!text.matches(
            String.format("for \\((%s )?%s = %s; %s %s %s; (\\+\\+%s|%s"
            + "\\+\\+|--%s|%s--)\\)",
                token, token, token, token, operator, token, token, 
                token, token, token))
           )
        {
            return new Token("For loop white space is incorrect", 
                yyline + 1, true);
        }
    }
    else // worst case scenario - for loop is not typical; does its 
         // best to detect missing whitespace
    {
        if (!text.matches("for \\(.*;( .*)?;( .*)?\\)"))
        {
            return new Token("For loop white space is incorrect", 
                yyline + 1, true);
        }
    }

    }

// if/while/switch construct
<PASS5> ^{WS}*(if|while|switch){WS}*\(  {
    String text = yytext();
    if (!text.matches("[ \\t\\f]*(if|while|switch) \\("))
    {
        return new Token("Construct should have one space between keyword and open parenthesis", yyline + 1, true);
    }
    }

// line before a for/while/if/switch statement
<PASS5> ^.*[^ \t\f\{\n\r].*{LT}/{WS}*(for|while|if|switch).*{LT}{WS}*\{ {
    return new Token("Missing whitespace unless this line is the "
        + "initializer for an accumulator variable", yyline + 1, false);
    }

// line after the end of a brace block
<PASS5> \}{EM}.*$ {
    //System.out.printf("%d - end brace met\n", yyline + 1);
    String text = yytext();
    String[] lines = text.split("\\r\\n|\\n");
    String lastLine = lines[lines.length - 1];
    lastLine = lastLine.split("//", 2)[0];
    //System.out.println("lastLine = " + lastLine);
    yypushback(Math.min(lastLine.length() + 1, yylength() - 1));
    if (!lastLine.matches("([} \\t\\f\\r\\n]*)|([ \t\f]*(else|catch)([ \t\f].*)?)"))
    {
        return new Token("Missing whitespace after brace", yyline + 1,
            true);
    }
    }

// white space before an open brace
<PASS5> ^({EM}{WS}*)/\{ {
    return new Token("Likely superfluous new line before brace",
        yyline + 1, false);
    }

// white space after an open brace
<PASS5> ^{WS}*\{{EM}{EM} {
    return new Token("Superfluous new line after brace", yyline + 2,
        true);
    }

// white space before a close brace
<PASS5> ^{EM}/{WS}*\} {
    String text = yytext();
    String[] lines = text.split("\\r\\n|\\n");
    //yypushback(1);
    return new Token("Superfluous new line before brace", yyline + 1,
        true);
    }

// class/interface definitions
<PASS5> ^.*{LT}{CL}   {
    isJava = true;
    //System.out.println("Class met");
    String text = yytext();
    String[] lines = text.split("\\r\\n|\\n", 2);
    String[] parts = lines[1].split("[ \t\f]+");
    String className = parts[parts.length - 1];
    if (!className.matches("([A-Z0-9_][a-z0-9_]*)*"))
    {
        return new Token("Class/interface name " + className 
            + " is not upper camel case", yyline + 1, true);
    }
    if (!lines[0].matches("^.*\\*\\/[ \\t\\f]*$"))
    {
        return new Token("Class/interface should be preceded by a block"
            + " comment unless it would clearer to put it before import"
            + " statements", yyline + 1, false);
    }
    }

// variable definitions (or close enough)
<PASS5> ^{WS}*(for{WS}+\()?((({ID}{WS}+)?{ID}{WS}+)?{ID}{WS}+)?{ID}{WS}+{ID}(;|{WS}+=) {
    String text = yytext();
    //System.out.println(text);
    String[] parts = text.split("[ \\t\\f\\(=;]+");
    //System.out.println(Arrays.toString(parts));
    String variableName = parts[parts.length - 1];

    boolean withinFor = (parts[0].equals("for") || 
        (parts[0].equals("") && parts[1].equals("for")));

    boolean constant = false;
    for (int i = 0; i < parts.length - 1; i++)
    {
        if (parts[i].equals("final") || parts[i].equals("#define"))
        {
            constant = true;
            break;
        }
    }
    

    if (variableName.equals("l") || variableName.equals("O"))
    {
        return new Token("Variable name " + variableName 
            + " is always invalid ", yyline + 1, true);
    }
    else if ((variableName.equals("i") || variableName.equals("j") || variableName.equals("k")) && !withinFor)
    {
        return new Token("Variable " + variableName 
            + " has potentially an invalid name because it is not a"
            + " loop constant unless it maps to a design document or"
            + " has a physical significance", yyline + 1, false);
    }
    else if (variableName.length() == 1 && 
        Character.isUpperCase(variableName.charAt(0)))
    {
        return new Token("Variable " + variableName + " has an invalid"
            + " name unless it maps to a design document or has a"
            + " physical significance", yyline + 1, true);
    }
    else if ((!variableName.matches("[a-z0-9]+([A-Z_][a-z0-9]*)*") &&
             !variableName.matches("[A-Z0-9_]*")) && !constant)
    {
        return new Token("Variable name " + variableName + " should be"
            + " lower camel case or upper snake case", yyline + 1, 
            true);
    }
    else if (constant && !variableName.matches("[A-Z0-9_]*"))
    {
        return new Token("Constant variable " + variableName + " should"
            + " be in upper snake case", yyline + 1, true);
    }
    }


<PASS5> {LT}|.  {}

<PASS5> <<EOF>> {
    System.out.println("Starting Pass 6");
    resetFileReading();
    yybegin(PASS6);
    return Token.NULL;
    }


<PASS6> (\/\*)~(\*\/) {
    String length = yytext();
    String[] lines = length.split("\\r\\n|\\n");
    yypushback(Math.min(lines[lines.length-1].length() + 2, yylength() - 1));
    }

<PASS6> \/\/.*$ {}

<PASS6> \"([^\"]|(\\\"))*\" {}

<PASS6> ^{WS}*(for|while).*{LT}{WS}*\{.*$   {
    //System.out.println("for/while loop met");
    //System.out.println(yytext());
    stack.push(2);
    //System.out.println(stack);
    }

<PASS6> ^{WS}*(switch).*{LT}{WS}*\{.*$ {
    //System.out.println("switch statement met");
    //System.out.println(yytext());
    stack.push(1);
    //System.out.println(stack);
    }

// if, else, else if
<PASS6> ^{WS}*(if{WS}*|else{WS}*).*{LT}{WS}*\{.*$ {
    //System.out.println("other statement met");
    //System.out.println(yytext());
    stack.push(0);
    //System.out.println(stack);
    }

// function/method header
<PASS6> ^.*{LT}({WS}{3})?{FN}.*$  {
    //System.out.println("function/method met");
    String text = yytext();
    String[] lines = text.split("\\r\\n|\\n", 2);
    text = lines[1].substring(0, lines[1].indexOf('('));
    String[] keywords = text.split("[ \t\f]+");
    //System.out.println(Arrays.toString(keywords));
    for (String s : keywords)
    {
        if (s.matches("if|else|class|interface"))
        {
            yypushback(Math.min(yylength() - lines[0].length(), yylength() - 1));
            return Token.NULL;
        }
    }
    //System.out.println(lines[0]);

    boolean isVoid = keywords[keywords.length - 2].equals("void");

    stack.push(isVoid ? 4 : 3);
    //System.out.println(stack);
    if (!lines[0].matches(".*\\*\\/[ \t\f]*"))
    {
        return new Token("Function/method must be immediately preceded"
            + " by a block comment", yyline + 1, true);
    }
    if (!text.matches("^.*[^ \t\f]$"))
    {
        return new Token("There should be no space between method/"
            + " function name and parentheses", yyline + 1, true);
    }
    }

// generic block statement
<PASS6> ^{WS}*.*{LT}{WS}*\{.*$   {
    //System.out.println("other statement met");
    //System.out.println(yytext());
    stack.push(0);
    //System.out.println(stack);
    }

// final return statement
<PASS6> ^{WS}*return({WS}.*|\(.*|{WS}*);{EM}+{WS}*\}    {
    //System.out.println("final return statement");
    if (!stack.empty())
    {
        if (stack.peek() == 3 || stack.peek() == 4)
        {
            stack.pop();
            //System.out.println(stack);
        }
        else
        {
            //System.out.println(stack);
            yypushback(yylength() - yytext().indexOf('r') - 1);
            return new Token("Return statement is not the last"
                + " executable line of the method/function. This is"
                + " only allowed for \"very small functions\"",
                yyline + 1, false);
        }
    }
    }

// default return statement
<PASS6> ^{WS}*return{WS}+.*;{EM}    {
    //System.out.println("default return statement met");
    //System.out.println(stack);
    return new Token("Return statement is not the last executable line"
        + " of the method/function. This is only allowed for \"very"
        + " small functions\"",
        yyline + 1, false);
    }

// generic close brace
<PASS6> ^{WS}*\}   {
    //System.out.println("close statement met");
    //System.out.println(yytext());
    if (!stack.empty())
    {
        if (stack.peek() == 3 || (stack.peek() == 4 && !isJava))
        {
            stack.pop();
            return new Token("Missing final return statement at end of"
                + " the method/function", yyline + 1, true);
        }
        stack.pop();
    }
    //System.out.println(stack);
    }

<PASS6> break   {
    //System.out.println("break met");
    //System.out.println(stack);
    if (!stack.empty() && stack.lastIndexOf(2) > stack.lastIndexOf(1))
    {
        return new Token("Break statement in loop", yyline + 1, true);
    }
    }


<PASS6> {LT}|.  {/*System.out.print(yytext());*/}

<PASS6> <<EOF>> {
    return new Token("EOF", -1, false);
    }
