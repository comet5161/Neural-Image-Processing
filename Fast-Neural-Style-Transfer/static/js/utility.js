var batch_download = function () {
    curr_idx = 0;
    batch_pic = [];
    $(".batch-box button").text("下载中... 1%").attr("disabled", true);
    $(".big_download:visible").each(function (i, o) {
        batch_pic.push([$(o).attr('download'), $(o).attr('href')])
    });
    zip.createWriter(new zip.BlobWriter(), function (writer) {
        add_pic(writer);
    });
};