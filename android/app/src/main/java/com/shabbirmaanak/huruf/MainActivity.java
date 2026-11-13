package com.shabbirmaanak.huruf;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(HurufSpeech.class);
        registerPlugin(HurufPrint.class);
        super.onCreate(savedInstanceState);
    }
}
