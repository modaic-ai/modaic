import re
import hashlib


def sanitize_name(original_name: str) -> str: # TODO: also sanitize SQL keywords
    """
    Sanitizes names of files and directories.
    
    Rules: 
    1. Remove file extension
    2. Replace illegal characters with underscores
    3. Replace consecutive consecutive underscores/illegal charachters with a single underscore
    4. Replace - with _
    5. no caps
    4. remove leading/trailing underscores
    5. if name starts with a number, add t_
    6. if name is longer than 64 chars, truncate and add a hash suffix
    
    Args:
        original_name: The name to sanitize. 
        
    Returns:
        The sanitized name.
    """
    # Remove file extension
    name = original_name.split('.')[0]
    
    # Replace illegal characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    if len(name) > 2:
        name = name.strip('_')
    
    # Convert to lowercase
    name = name.lower()
    
    # Ensure name does not start with a number
    if name[0].isdigit():
        name = 't_' + name
    
    # If name is longer than 64 chars, truncate and add a hash suffix
    if len(name) > 64:
        prefix = name[:20].rstrip('_')
        hash_suffix = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        name = f"{prefix}_{hash_suffix}"
    
    return name

def is_valid_table_name(name: str) -> bool:
    """
    Checks if a name is a valid table name.
    
    Args:
        name: The name to validate.
        
    Returns:
        True if the name is valid, False otherwise.
    """
    valid = (name.islower() and 
             not name.startswith('_') and 
             not name.endswith('_') and
             not name[0].isdigit()
             and len(name) <= 64
             )
    return valid