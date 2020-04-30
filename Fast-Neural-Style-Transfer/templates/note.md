

对于长期占用google服务器的计算资源，有可能会断开连接，这样的话，如果模型在晚上进行训练，自动断开的化就会降低效率，这里可以考虑加个自动检测的，当检测到重新连接的按钮，就尝试点击该按钮，代码如下：
在训练页面下点击右键/检查/console/  下粘贴一下代码即可，下面设置的是2分钟检查一次

function testConnect(){
	console.log(new Date())
	var btn = document.getElementsByClassName('colab-toolbar-button');
 
	for(var i=0; i<btn.length; i++){
		var txt1 = btn[i].innerHTML.trim().indexOf('重新连接');
		var txt2 = btn[i].innerHTML.trim().indexOf('连接');
		if(txt1 === 0 || txt2 === 0){
			console.log('点击 连接 按钮');
			btn[i].click();
			break;
		}
	}
}
 
var myTimer = setInterval(testConnect, 120000)
 
——————————————

如果你还不知道Colab，那一定要体验一下，这个能在线编程、还能白嫖Google云TPU/GPU训练自己AI模型的工具早已圈了一大波粉丝。
但是，作为白嫖的福利，它总有限制，比如你不去碰它，过30分钟Colab就会自动掉线。
所以，程序员ShIvam Rawat在medium上贴出了一段代码：

function ClickConnect{
console.log(“Working”);
document.querySelector(“colab-toolbar-button #connect”).click
}
setInterval(ClickConnect, 60000)

你只要把它放进控制台，它就会自动隔一阵儿调戏一下Colab页面，防止链接断掉。

是不是非常机智？