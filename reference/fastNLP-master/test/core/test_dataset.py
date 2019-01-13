import os
import unittest

from fastNLP.core.dataset import DataSet
from fastNLP.core.fieldarray import FieldArray
from fastNLP.core.instance import Instance


class TestDataSet(unittest.TestCase):

    def test_init_v1(self):
        ds = DataSet([Instance(x=[1, 2, 3, 4], y=[5, 6])] * 40)
        self.assertTrue("x" in ds.field_arrays and "y" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["x"].content, [[1, 2, 3, 4], ] * 40)
        self.assertEqual(ds.field_arrays["y"].content, [[5, 6], ] * 40)

    def test_init_v2(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        self.assertTrue("x" in ds.field_arrays and "y" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["x"].content, [[1, 2, 3, 4], ] * 40)
        self.assertEqual(ds.field_arrays["y"].content, [[5, 6], ] * 40)

    def test_init_assert(self):
        with self.assertRaises(AssertionError):
            _ = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 100})
        with self.assertRaises(AssertionError):
            _ = DataSet([[1, 2, 3, 4]] * 10)
        with self.assertRaises(ValueError):
            _ = DataSet(0.00001)

    def test_append(self):
        dd = DataSet()
        for _ in range(3):
            dd.append(Instance(x=[1, 2, 3, 4], y=[5, 6]))
        self.assertEqual(len(dd), 3)
        self.assertEqual(dd.field_arrays["x"].content, [[1, 2, 3, 4]] * 3)
        self.assertEqual(dd.field_arrays["y"].content, [[5, 6]] * 3)

    def test_add_append(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.add_field("z", [[5, 6]] * 10)
        self.assertEqual(len(dd), 10)
        self.assertEqual(dd.field_arrays["x"].content, [[1, 2, 3]] * 10)
        self.assertEqual(dd.field_arrays["y"].content, [[1, 2, 3, 4]] * 10)
        self.assertEqual(dd.field_arrays["z"].content, [[5, 6]] * 10)

        with self.assertRaises(RuntimeError):
            dd.add_field("??", [[1, 2]] * 40)

    def test_delete_field(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.delete_field("x")
        self.assertFalse("x" in dd.field_arrays)
        self.assertTrue("y" in dd.field_arrays)

    def test_getitem(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ins_1, ins_0 = ds[0], ds[1]
        self.assertTrue(isinstance(ins_1, Instance) and isinstance(ins_0, Instance))
        self.assertEqual(ins_1["x"], [1, 2, 3, 4])
        self.assertEqual(ins_1["y"], [5, 6])
        self.assertEqual(ins_0["x"], [1, 2, 3, 4])
        self.assertEqual(ins_0["y"], [5, 6])

        sub_ds = ds[:10]
        self.assertTrue(isinstance(sub_ds, DataSet))
        self.assertEqual(len(sub_ds), 10)

    def test_get_item_error(self):
        with self.assertRaises(RuntimeError):
            ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
            _ = ds[40:]

        with self.assertRaises(KeyError):
            ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
            _ = ds["kom"]

    def test_len_(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        self.assertEqual(len(ds), 40)

        ds = DataSet()
        self.assertEqual(len(ds), 0)

    def test_apply(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.apply(lambda ins: ins["x"][::-1], new_field_name="rx")
        self.assertTrue("rx" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["rx"].content[0], [4, 3, 2, 1])

        ds.apply(lambda ins: len(ins["y"]), new_field_name="y")
        self.assertEqual(ds.field_arrays["y"].content[0], 2)

        res = ds.apply(lambda ins: len(ins["x"]))
        self.assertTrue(isinstance(res, list) and len(res) > 0)
        self.assertTrue(res[0], 4)

    def test_drop(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6], [7, 8, 9, 0]] * 20})
        ds.drop(lambda ins: len(ins["y"]) < 3)
        self.assertEqual(len(ds), 20)

    def test_contains(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        self.assertTrue("x" in ds)
        self.assertTrue("y" in ds)
        self.assertFalse("z" in ds)

    def test_rename_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.rename_field("x", "xx")
        self.assertTrue("xx" in ds)
        self.assertFalse("x" in ds)

        with self.assertRaises(KeyError):
            ds.rename_field("yyy", "oo")

    def test_input_target(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        self.assertTrue(ds.field_arrays["x"].is_input)
        self.assertTrue(ds.field_arrays["y"].is_target)

        with self.assertRaises(KeyError):
            ds.set_input("xxx")
        with self.assertRaises(KeyError):
            ds.set_input("yyy")

    def test_get_input_name(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        self.assertEqual(ds.get_input_name(), [_ for _ in ds.field_arrays if ds.field_arrays[_].is_input])

    def test_get_target_name(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        self.assertEqual(ds.get_target_name(), [_ for _ in ds.field_arrays if ds.field_arrays[_].is_target])

    def test_apply2(self):
        def split_sent(ins):
            return ins['raw_sentence'].split()

        dataset = DataSet.read_csv('test/data_for_tests/tutorial_sample_dataset.csv', headers=('raw_sentence', 'label'),
                                   sep='\t')
        dataset.drop(lambda x: len(x['raw_sentence'].split()) == 0)
        dataset.apply(split_sent, new_field_name='words', is_input=True)
        # print(dataset)

    def test_add_field(self):
        ds = DataSet({"x": [3, 4]})
        ds.add_field('y', [['hello', 'world'], ['this', 'is', 'a', 'test']], is_input=True, is_target=True)
        # ds.apply(lambda x:[x['x']]*3, is_input=True, is_target=True, new_field_name='y')
        print(ds)

    def test_save_load(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.save("./my_ds.pkl")
        self.assertTrue(os.path.exists("./my_ds.pkl"))

        ds_1 = DataSet.load("./my_ds.pkl")
        os.remove("my_ds.pkl")

    def test_get_all_fields(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ans = ds.get_all_fields()
        self.assertEqual(ans["x"].content, [[1, 2, 3, 4]] * 10)
        self.assertEqual(ans["y"].content, [[5, 6]] * 10)

    def test_get_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ans = ds.get_field("x")
        self.assertTrue(isinstance(ans, FieldArray))
        self.assertEqual(ans.content, [[1, 2, 3, 4]] * 10)
        ans = ds.get_field("y")
        self.assertTrue(isinstance(ans, FieldArray))
        self.assertEqual(ans.content, [[5, 6]] * 10)

    def test_reader(self):
        # 跑通即可
        ds = DataSet().read_naive("test/data_for_tests/tutorial_sample_dataset.csv")
        self.assertTrue(isinstance(ds, DataSet))
        self.assertTrue(len(ds) > 0)

        ds = DataSet().read_rawdata("test/data_for_tests/people_daily_raw.txt")
        self.assertTrue(isinstance(ds, DataSet))
        self.assertTrue(len(ds) > 0)

        ds = DataSet().read_pos("test/data_for_tests/people.txt")
        self.assertTrue(isinstance(ds, DataSet))
        self.assertTrue(len(ds) > 0)


class TestDataSetIter(unittest.TestCase):
    def test__repr__(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        for iter in ds:
            self.assertEqual(iter.__repr__(), "{'x': [1, 2, 3, 4],\n'y': [5, 6]}")
