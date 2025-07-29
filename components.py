if __name__ == '__main__':
    import cci
    
    df = cci.i_o.load_dialogues("cci/data/data.pkl")
    
    try:
        # Compute CCI scores with component analysis (single-threaded to avoid Windows multiprocessing issues)
        result = cci.pipeline.compute_cci_with_components(df, use_multiprocessing=False)
        print("SUCCESS: Component analysis complete!")
        print(f"Total speaker pairs analyzed: {len(result)}")
        
        # Show first few results
        print("\nFirst 3 results:")
        print(result[['dialogue_id', 'from_speaker', 'to_speaker', 'CCI_score', 'D_mean', 'I_mean', 'S_mean']].head(3))
        
        # Analyze component contributions
        non_zero = result[result['CCI_score'] != 0]
        if len(non_zero) > 0:
            print(f"\nComponent analysis for {len(non_zero)} non-zero interactions:")
            print(f"D_t (Divergence) - Mean: {non_zero['D_mean'].mean():.4f}, Std: {non_zero['D_std'].mean():.4f}")
            print(f"I_t+1 (Incorporation) - Mean: {non_zero['I_mean'].mean():.4f}, Std: {non_zero['I_std'].mean():.4f}")
            print(f"S_t+1 (Shared Growth) - Mean: {non_zero['S_mean'].mean():.4f}, Std: {non_zero['S_std'].mean():.4f}")
            
            # Show component ranges
            print(f"\nComponent ranges:")
            print(f"D_t: [{non_zero['D_mean'].min():.4f}, {non_zero['D_mean'].max():.4f}]")
            print(f"I_t+1: [{non_zero['I_mean'].min():.4f}, {non_zero['I_mean'].max():.4f}]")
            print(f"S_t+1: [{non_zero['S_mean'].min():.4f}, {non_zero['S_mean'].max():.4f}]")
        else:
            print("\nNo non-zero CCI scores found for component analysis.")
            
    except Exception as e:
        print(f"Error computing CCI: {e}")
        import traceback
        traceback.print_exc()