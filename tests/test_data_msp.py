from atom3d_menagerie.data.msp import parse_tag, Tag

def test_parse_tag_1acb():
    tag = parse_tag('1ACB_E_I_LI38D')
    assert tag == Tag(
            entry_id='1ACB',
            chain_1='E',
            chain_2='I',
            resn_wt='L',
            chain_mut='I',
            resi=38,
            resn_mut='D',
    )

def test_parse_tag_1jrh():
    tag = parse_tag('1JRH_LH_I_DL28A')
    assert tag == Tag(
            entry_id='1JRH',
            chain_1='LH',
            chain_2='I',
            resn_wt='D',
            chain_mut='L',
            resi=28,
            resn_mut='A',
    )




