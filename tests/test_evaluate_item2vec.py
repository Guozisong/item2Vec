import pandas as pd

from item2vec.evaluation import evaluate_predictions, load_item_categories, split_orders


def test_split_orders_keeps_complete_orders_in_one_period():
    rows = pd.DataFrame([
        ('old', 'A', '20261001'), ('old', 'B', '20261002'),
        ('middle', 'A', '20261010'), ('middle', 'C', '20261010'),
        ('new', 'A', '20261020'), ('new', 'D', '20261020'),
    ], columns=['order_id', 'prod_id', 'dt'])

    train, validation, test = split_orders(rows, validation_start='20261008',
                                             test_start='20261015')

    assert train.order_id.tolist() == ['old', 'old']
    assert validation.order_id.tolist() == ['middle', 'middle']
    assert test.order_id.tolist() == ['new', 'new']


def test_split_orders_assigns_cross_boundary_order_to_later_period():
    rows = pd.DataFrame([
        ('one', 'A', '20261007'), ('one', 'B', '20261008'),
    ], columns=['order_id', 'prod_id', 'dt'])

    train, validation, test = split_orders(rows, validation_start='20261008',
                                             test_start='20261015')

    assert train.empty and test.empty
    assert validation.order_id.tolist() == ['one', 'one']


def test_evaluation_counts_heldout_pairs_without_using_them_for_ranking():
    orders = pd.DataFrame([
        ('o1', 'A'), ('o1', 'B'), ('o1', 'B'),
        ('o2', 'A'), ('o2', 'B'), ('o3', 'A'), ('o3', 'C'),
    ], columns=['order_id', 'prod_id'])
    predictions = pd.DataFrame([
        ('A', 'B', .9), ('A', 'D', .8), ('B', 'A', .7),
    ], columns=['master_prod_id', 'slave_prod_id', 'similarity'])

    metrics = evaluate_predictions(orders, predictions, top_k=1)

    assert metrics['pair_recall_at_k'] == 2 / 4
    assert metrics['weighted_recall_at_k'] == 4 / 6
    assert metrics['source_coverage'] == 2 / 3


def test_category_metrics_include_missing_predictions_and_skip_unknown_category():
    orders = pd.DataFrame([
        ('o1', 'A'), ('o1', 'B'), ('o1', 'C'),
        ('o2', 'A'), ('o2', 'C'), ('o3', 'A'), ('o3', 'D'),
    ], columns=['order_id', 'prod_id'])
    predictions = pd.DataFrame([
        ('A', 'B'), ('A', 'C'), ('B', 'A'),
    ], columns=['master_prod_id', 'slave_prod_id'])
    categories = {'A': '蛋', 'B': '蛋', 'C': '蔬菜'}

    metrics = evaluate_predictions(orders, predictions, top_k=2,
                                   item_categories=categories)

    assert metrics['same_category_pair_count'] == 2
    assert metrics['same_category_weighted_recall_at_k'] == 1
    assert metrics['cross_category_pair_count'] == 6
    assert metrics['cross_category_weighted_recall_at_k'] == 2 / 6


def test_category_metrics_return_none_when_no_known_pairs():
    orders = pd.DataFrame([('o1', 'A'), ('o1', 'B')], columns=['order_id', 'prod_id'])
    predictions = pd.DataFrame(columns=['master_prod_id', 'slave_prod_id'])

    metrics = evaluate_predictions(orders, predictions, item_categories={'A': '蛋'})

    assert metrics['same_category_weighted_recall_at_k'] is None
    assert metrics['cross_category_weighted_recall_at_k'] is None


def test_load_item_categories_uses_text_before_brand(tmp_path):
    path = tmp_path / 'item.csv'
    path.write_text('prod_id,prod_description\n'
                    'A,"{蛋类中的鲜蛋类，商品品牌为甲，商品名称为鸡蛋}"\n'
                    'B,"{蔬菜中的叶菜类，商品品牌为乙，商品名称为青菜}"\n',
                    encoding='utf-8')

    assert load_item_categories(path) == {'A': '蛋类中的鲜蛋类', 'B': '蔬菜中的叶菜类'}
