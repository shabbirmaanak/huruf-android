package com.shabbirmaanak.huruf;

import android.content.Context;
import android.print.PrintAttributes;
import android.print.PrintManager;
import android.webkit.WebView;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/* Opens Android's print screen (print to a printer or save as PDF) for the page as it is now.
   The page puts the printable sheets in #printroot; its print styles hide everything else. */
@CapacitorPlugin(name = "HurufPrint")
public class HurufPrint extends Plugin {
    @PluginMethod
    public void print(PluginCall call) {
        String title = call.getString("title", "Huruf sheets");
        getActivity().runOnUiThread(() -> {
            try {
                WebView wv = getBridge().getWebView();
                PrintManager pm = (PrintManager) getActivity().getSystemService(Context.PRINT_SERVICE);
                PrintAttributes attrs = new PrintAttributes.Builder().setMediaSize(PrintAttributes.MediaSize.ISO_A4).build();
                pm.print(title, wv.createPrintDocumentAdapter(title), attrs);
                call.resolve();
            } catch (Exception e) {
                call.reject("print failed: " + e.getMessage());
            }
        });
    }
}
