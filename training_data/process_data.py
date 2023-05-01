import csv

def split_into_paragraphs(text, author, max_length=20):
    # Split text into words
    text_without_quotes = text.replace('"', '')
    words = text_without_quotes.split()
    print(words)
    
    # Create list of formatted paragraphs
    formatted_paragraphs = []
    i = 0
    while i < len(words):
        # Get the next max_length words
        paragraph_words = words[i:i+max_length]
        
        # Join the words into a paragraph string
        paragraph = ' '.join(paragraph_words)
        
        # Format paragraph as quoted string
        formatted_p = f"id{i//max_length+1000}-{paragraph}-{author}"
        formatted_paragraphs.append(formatted_p)
        
        # Move to the next max_length words
        i += max_length
    
    return formatted_paragraphs

# Input file path
input_file = 'a_gorman_validation.txt'

# Output file path
output_file = 'm_angelou.csv'

# Author name
author = 'MA'

# Read input file
with open(input_file, 'r') as f:
    text = f.read()
    #print(text)

# Split text into paragraphs and format them
formatted_paragraphs = split_into_paragraphs(text, author)
#print(formatted_paragraphs)
# Write formatted paragraphs to CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow([str("id"), 'text', 'author'])
    for p in formatted_paragraphs:
        print(p)
        writer.writerow([field for field in p.split('-')])
print('Done!')

