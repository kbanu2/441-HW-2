package TextProcessing;


/**
 * Generated from IDL interface "FileProcessor".
 *
 * @author JacORB IDL compiler V 3.9
 * @version generated at Oct 5, 2024, 2:04:38 PM
 */

public interface FileProcessorOperations
{
	/* constants */
	/* operations  */
	TextProcessing.ProcessingResult processChunk(java.lang.String chunkContent);
	void writeResults(TextProcessing.ProcessingResult results);
}
