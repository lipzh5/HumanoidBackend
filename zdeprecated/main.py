# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import asyncio
from AmecaSubRouter import SubRouter


async def run_sub_router():
	sub_router = SubRouter()
	loop = asyncio.get_event_loop()
	
	task1 = loop.create_task(sub_router.sub_vcap_data())
	task2 = loop.create_task(sub_router.route_visual_task())
	# task3 = loop.create_task(sub_router.sub_face_detect_data())
	await asyncio.gather(task1, task2)



if __name__ == "__main__":
    asyncio.run(run_sub_router())