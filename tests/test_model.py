from app.model import train_and_predict

def test_predictions_not_none():
    preds, _, _ = train_and_predict()
    assert preds is not None

def test_predictions_length():
    preds, y_test, _ = train_and_predict()
    assert len(preds) == len(y_test)
    assert len(preds) > 0

def test_predictions_value_range():
    preds, _, _ = train_and_predict()
    assert all(p in [0, 1] for p in preds)

def test_model_accuracy():
    _, _, acc = train_and_predict()
    assert acc >= 0.7
