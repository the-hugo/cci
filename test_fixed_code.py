if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import cci
    
    print('Testing fixed CCI implementation...')
    
    # Test concept extraction
    try:
        concepts = cci.concepts.extract_concepts("The dog runs quickly in the park")
        print(f'Concept extraction works: {concepts}')
    except Exception as e:
        print(f'Concept extraction failed: {e}')
        import traceback
        traceback.print_exc()
        
    # Test semantic similarity
    try:
        sim = cci.metrics.semantic_similarity("dog", "puppy")
        print(f'Semantic similarity works: dog-puppy = {sim:.3f}')
    except Exception as e:
        print(f'Semantic similarity failed: {e}')
        import traceback
        traceback.print_exc()
        
    # Test incorporation
    try:
        novel = {"dog", "happy"}
        future = {"puppy", "joy", "cat"}
        incorp = cci.metrics.incorporation(novel, future)
        print(f'Incorporation works: {incorp:.3f}')
    except Exception as e:
        print(f'Incorporation failed: {e}')
        import traceback
        traceback.print_exc()
        
    print('\nFixed implementation test complete!')