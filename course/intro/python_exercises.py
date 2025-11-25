def sum_list(numbers):
    """Given a list of integers 'numbers'
    return the sum of this list."""
    return sum(numbers)


def max_value(numbers):
    """Given a list of numbers 'numbers'
    return the maximum value of this list."""
    return max(numbers)


def reverse_string(s):
    """Given a string 'string'
    return the reversed version of the input string."""
  
    rev_string = s[::-1]
    return rev_string


def filter_even(numbers):
    """Given a list of numbers 'numbers'
    return a list containing only the even numbers from the input list."""
    even = []
    for num in numbers:
      if num % 2 == 0:
        even.append(num)
    return even


def get_fifth_row(df):
    """Given a dataframe 'df'
    return the fifth row of this as a pandas DataFrame."""
    return df.iloc[4]


def column_mean(df, column):
    """Given a dataframe 'df' and the name of a column 'column'
    return the mean of the specified column in a pandas DataFrame."""
    mean_col = df[column].mean()
    return mean_col


def lookup_key(d, key):
    """Given a dictionary 'd' and a key 'key'
    return the value associated with the key in the dictionary."""
    return d.get(key)


def count_occurrences(lst):
    """Given a list 'lst'
    return a dictionary with counts of each unique element in the list."""
    counts = dict()
    for element in lst:
      counts[element] = counts.get(element, 0) + 1
    return counts


def drop_missing(df):
    """Given a dataframe 'df' with some rows containing missing values,
    return a DataFrame with rows containing missing values removed."""
    return df.dropna()


def value_counts_df(df, column):
    """Given a dataframe 'df' with various columns and the name of one of those columns 'column',
    return a DataFrame with value counts of the specified column."""
    counts = df[column].value_counts().reset_index()
    return counts
