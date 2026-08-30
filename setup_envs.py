import grid2op

if __name__ == "__main__":
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)
    env.train_val_split_random(add_for_train="train", add_for_test="test", add_for_val="val", pct_val=5., pct_test=5.)
    print("14 bus system download complete!")
    env_name = "l2rpn_wcci_2020"
    env = grid2op.make(env_name)
    env.train_val_split_random(add_for_train="train", add_for_test="test", add_for_val="val", pct_val=5., pct_test=5.)
    print("36 bus system download complete!")

    #env_name = "l2rpn_wcci_2022"
    #env = grid2op.make(env_name)
    #env.train_val_split_random(add_for_train="train", add_for_test="test", add_for_val="val", pct_val=5., pct_test=5.)
